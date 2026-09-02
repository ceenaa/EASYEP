"""Tests for the GPU-free logic in easyep_v4.py.

There is deliberately no test of answer quality: the runs emit completions only,
and grading is done afterwards by an external judge.

Everything here runs on a login node in seconds. It covers the parts a reviewer
would want to check by reading -- mask construction, the control baselines, the
discrimination metric, blinding -- so those can be verified without a 164 GiB
model load.

    python test_easyep_v4.py
"""
import hashlib
import hmac
import importlib.util
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("e", HERE / "easyep_v4.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
provenance_spec = importlib.util.spec_from_file_location(
    "checkpoint_provenance", HERE / "checkpoint_provenance.py")
C = importlib.util.module_from_spec(provenance_spec)
provenance_spec.loader.exec_module(C)

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  ok    {name}")
    except AssertionError as exc:
        FAIL.append((name, str(exc)))
        print(f"  FAIL  {name}: {exc}")
    except Exception as exc:  # noqa: BLE001
        FAIL.append((name, f"{type(exc).__name__}: {exc}"))
        print(f"  ERROR {name}: {type(exc).__name__}: {exc}")


def must_raise(exc_type, fn, contains=""):
    """Small dependency-free equivalent of pytest.raises for this test runner."""
    try:
        fn()
    except exc_type as exc:
        if contains:
            assert contains.lower() in str(exc).lower(), str(exc)
        return exc
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"expected {exc_type.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"expected {exc_type.__name__}, but no exception was raised")


N_L, N_E, KEEP, N_HASH = 43, 256, 128, 3


def fixture_model_identity(fill="0"):
    digest = fill * 64
    return {
        "identity_schema_version": E.MODEL_IDENTITY_SCHEMA_VERSION,
        "instrumentation_sha256": digest,
        "config": {"name": "config.json", "sha256": digest},
        "official_sources": {
            "inference/model.py": digest,
            "inference/generate.py": digest,
            "encoding/encoding_dsv4.py": digest,
        },
        "tokenizer": {"tokenizer.json": digest},
        "checkpoint": {
            "world_size": 1,
            "shards": [{
                "name": "model0-mp1.safetensors",
                "size": 123,
                "sample_sha256": digest,
            }],
        },
        "runtime": {
            "python": "test", "torch": "test", "cuda": "test",
            "transformers": "test", "safetensors": "test",
        },
    }


def fixture_calibration_provenance():
    digest = "a" * 64
    return {
        "provenance_schema_version": 1,
        "source_kind": "corpus_files", "security_prompt": True,
        "source_samples": 1, "chunks": 1, "forwards": 1, "response_tokens": 1,
        "parameters": {"seed": 965, "temperature": 1.0, "max_seq_len": 64,
                       "max_new_tokens": 8, "max_chunks": 0},
        "selected_sources": [{"file_index": 0, "path": "sample.js",
                              "characters": 12, "text_sha256": digest}],
        "prompt_chunks": [{"file_index": 0, "chunk_index": 0, "n_chunks": 1,
                           "path": "sample.js", "prompt_tokens": 16,
                           "prompt_ids_sha256": digest}],
        "generated_responses": [{"run_index": 0, "path": "sample.js",
                                 "file_index": 0, "chunk_index": 0, "tokens": 1,
                                 "token_ids_sha256": digest}],
    }


def t_cutoff_diagnostics_expose_an_arbitrary_tie_break():
    """An integer ranking ties at the cut and topk resolves it by index."""
    n_layers, n_experts, keep, n_hash = 3, 8, 4, 1
    # every expert selected the same number of times: the cut is pure index order
    tied = torch.ones((n_layers, n_experts), dtype=torch.int64)
    report = E.score_cutoff_diagnostics(tied, keep, n_hash)
    assert report["layers_with_arbitrary_cut"] == n_layers - n_hash
    assert report["max_experts_tied_at_cutoff"] == n_experts
    for row in report["per_layer"]:
        assert row["absolute_margin"] == 0.0 and row["cut_is_arbitrary"] is True

    # a strictly ordered score has a real margin and no tie
    distinct = torch.arange(n_layers * n_experts, dtype=torch.float64).reshape(
        n_layers, n_experts)
    clean = E.score_cutoff_diagnostics(distinct, keep, n_hash)
    assert clean["layers_with_arbitrary_cut"] == 0
    assert clean["max_experts_tied_at_cutoff"] == 1
    for row in clean["per_layer"]:
        assert row["absolute_margin"] > 0 and row["cut_is_arbitrary"] is False


def t_calibration_loader_refuses_to_guess_at_text():
    """A record with no text field must stop the run, not become a JSON dump."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ok = root / "ok.json"
        ok.write_text(json.dumps(["plain text", {"text": "from text"},
                                  {"prompt": "from prompt"},
                                  {"content": "from content"}]))
        assert E.load_calibration(ok) == [
            "plain text", "from text", "from prompt", "from content"]

        # the real questions file shape: rich records, no text/prompt/content
        leaky = root / "questions.json"
        leaky.write_text(json.dumps([
            {"id": "Q01", "snippet": "var x=1;", "vulnerability": "XSS",
             "expected_reasoning": "attacker controls x"}]))
        must_raise(SystemExit, lambda: E.load_calibration(leaky), "JSON dump")

        lines = root / "rows.jsonl"
        lines.write_text(json.dumps({"id": 1, "snippet": "x"}) + "\n")
        must_raise(SystemExit, lambda: E.load_calibration(lines), "JSON dump")

        empty = root / "empty.json"
        empty.write_text(json.dumps([{"text": ""}]))
        must_raise(SystemExit, lambda: E.load_calibration(empty), "no non-empty")


def t_calibration_provenance_records_its_distribution():
    """A score artifact must say which calibration distribution produced it.

    cmd_pairs excludes profiled files from matched-pair selection using the
    recorded source paths. An artifact profiled from sample texts records
    synthetic sample-N names that match nothing, so the exclusion would no-op
    and the run would report zero overlap while reusing a profiled program.
    """
    good = fixture_calibration_provenance()
    E._validate_calibration_provenance(good)          # accepted

    for field, bad_value in (("source_kind", "something_else"),
                             ("security_prompt", "yes")):
        broken = json.loads(json.dumps(good))
        broken[field] = bad_value
        must_raise(SystemExit,
                   lambda b=broken: E._validate_calibration_provenance(b),
                   "re-profile")
        missing = json.loads(json.dumps(good))
        del missing[field]
        must_raise(SystemExit,
                   lambda m=missing: E._validate_calibration_provenance(m),
                   "re-profile")

    # every evaluation consumer rejects a wrong-distribution artifact, not just
    # the one where the symptom was first noticed
    raw = json.loads(json.dumps(good)); raw["security_prompt"] = False
    for need_paths in (False, True):
        must_raise(SystemExit,
                   lambda n=need_paths: E._require_evaluation_calibration(
                       {"calibration": raw}, "fixture", need_corpus_paths=n),
                   "different distribution")
    samples = json.loads(json.dumps(good)); samples["source_kind"] = "sample_texts"
    # usable for the question evaluation, not for matched-pair exclusion
    assert E._require_evaluation_calibration(
        {"calibration": samples}, "fixture") is samples
    must_raise(SystemExit,
               lambda: E._require_evaluation_calibration(
                   {"calibration": samples}, "fixture", need_corpus_paths=True),
               "no corpus paths")
    must_raise(SystemExit,
               lambda: E._require_evaluation_calibration({}, "fixture"),
               "no calibration provenance")

    # the builder refuses an unknown kind rather than recording it
    must_raise(ValueError, lambda: E._calibration_provenance(
        [], [], {"forwards": 0, "response_tokens": 0, "generated_responses": []},
        seed=1, temperature=1.0, max_seq_len=64, max_new_tokens=8, max_chunks=0,
        source_kind="guess", security_prompt=True), "source_kind")


def t_pair_loading_excludes_calibration_inputs_and_path_escapes():
    with tempfile.TemporaryDirectory() as td:
        inputs = Path(td) / "inputs"
        root = inputs / "vulnerable-js-files"
        root.mkdir(parents=True)
        rows = []
        for stem in ("a", "b"):
            vulnerable = root / f"{stem}.js"
            secure = root / f"{stem}_Code.js"
            vulnerable.write_text(f"dangerous('{stem}')\n", encoding="utf-8")
            secure.write_text(f"safe('{stem}')\n", encoding="utf-8")
            rows.append({
                "original_file": f"vulnerable-js-files/{vulnerable.name}",
                "secure_file": f"vulnerable-js-files/{secure.name}",
                "alert_locations": 1,
            })
        manifest = root / "manifest.jsonl"
        manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        stats = {}
        picked = E.load_pairs(manifest, root, 1, seed=0, stats=stats,
                              exclude_relative_paths={"a.js"})
        assert picked[0]["vuln_path"].endswith("b.js")
        assert stats["skipped_calibration_overlap"] == 1

        outside = Path(td) / "outside.js"
        outside.write_text("not in the dataset\n", encoding="utf-8")
        escaped = root / "escaped.jsonl"
        escaped.write_text(json.dumps({
            "original_file": "../outside.js",
            "secure_file": "../outside.js",
            "alert_locations": 1,
        }) + "\n")
        must_raise(ValueError, lambda: E.load_pairs(escaped, root, 1), "escapes")


def t_checkpoint_provenance_create_and_verify():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        snapshot, checkpoint = root / "snapshot", root / "checkpoint"
        (snapshot / "inference").mkdir(parents=True)
        (snapshot / "encoding").mkdir(parents=True)
        checkpoint.mkdir()
        sources = {
            "inference/convert.py": "# converter\n",
            "inference/model.py": "# model\n",
            "inference/kernel.py": "# kernel\n",
            "inference/generate.py": "# generator\n",
            "encoding/encoding_dsv4.py": "# encoder\n",
        }
        for relative, content in sources.items():
            path = snapshot / relative
            path.write_text(content, encoding="utf-8")
        (snapshot / "inference/config.json").write_bytes(
            (HERE / "config_v4_flash.json").read_bytes())

        metadata_root = snapshot / ".cache/huggingface/download"
        for relative in (*sources, "inference/config.json"):
            metadata = metadata_root / f"{relative}.metadata"
            metadata.parent.mkdir(parents=True, exist_ok=True)
            etag = C.git_blob_sha1(snapshot / relative)
            metadata.write_text(
                f"{C.PINNED_REVISION}\n{etag}\n{time.time()}\n", encoding="utf-8")
        original_weight = snapshot / "model-00001-of-00001.safetensors"
        original_weight.write_bytes(b"synthetic original weight")
        weight_metadata = metadata_root / f"{original_weight.name}.metadata"
        weight_metadata.write_text(
            f"{C.PINNED_REVISION}\n{C.sha256_file(original_weight)}\n{time.time()}\n",
            encoding="utf-8")

        shard = checkpoint / "model0-mp1.safetensors"
        shard.write_bytes(b"synthetic checkpoint bytes")
        common = {
            "checkpoint": str(checkpoint), "model_id": C.PINNED_MODEL_ID,
            "model_revision": C.PINNED_REVISION,
            "inference_revision": C.PINNED_REVISION,
            "config": str(HERE / "config_v4_flash.json"), "nproc": 1,
        }

        race_checkpoint = root / "race-checkpoint"
        race_checkpoint.mkdir()
        race_shard = race_checkpoint / "model0-mp1.safetensors"
        race_shard.write_bytes(b"original race-checkpoint bytes")
        original_observation = C.sha256_file_observation

        def replace_after_observation(path, *args, **kwargs):
            result = original_observation(path, *args, **kwargs)
            if Path(path).resolve() == race_shard.resolve():
                replacement = race_checkpoint / "replacement.tmp"
                replacement.write_bytes(b"replacement checkpoint bytes")
                os.replace(replacement, race_shard)
            return result

        try:
            C.sha256_file_observation = replace_after_observation
            race_args = {**common, "checkpoint": str(race_checkpoint)}
            must_raise(
                SystemExit,
                lambda: C.create(SimpleNamespace(
                    **race_args, source_snapshot=str(snapshot), force=False)),
                "changed before provenance publication",
            )
        finally:
            C.sha256_file_observation = original_observation
        assert not (race_checkpoint / C.DEFAULT_NAME).exists()

        C.create(SimpleNamespace(**common, source_snapshot=str(snapshot), force=False))
        manifest = checkpoint / C.DEFAULT_NAME
        record = json.loads(manifest.read_text(encoding="utf-8"))
        assert record["checkpoint"]["shards"][0]["sha256"] == C.sha256_file(shard)
        assert record["content_identity_sha256"] == C.checkpoint_content_identity(record)
        relocated = json.loads(json.dumps(record))
        relocated["created_utc"] = "2099-01-01T00:00:00Z"
        relocated["checkpoint"]["root_at_creation"] = "/another/filesystem/checkpoint"
        relocated["checkpoint"]["shards"][0]["mtime_ns"] += 123
        assert (C.checkpoint_content_identity(relocated)
                == record["content_identity_sha256"]), \
            "semantic checkpoint identity must ignore paths, time, and mtime"
        # Normal launches need the attested runtime/conversion sources, not a
        # second retained copy of the enormous original HF weight shards.
        original_weight.unlink()
        verify_args = {**common, "manifest": "", "source_snapshot": str(snapshot)}
        C.verify(SimpleNamespace(**verify_args, full=False))
        C.verify(SimpleNamespace(**verify_args, full=True))

        stat = shard.stat()
        os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        must_raise(SystemExit,
                   lambda: C.verify(SimpleNamespace(**verify_args, full=False)),
                   "mtime")
        C.verify(SimpleNamespace(**verify_args, full=True))

        shard.write_bytes(b"corrupted checkpoint bytes")
        must_raise(SystemExit,
                   lambda: C.verify(SimpleNamespace(**verify_args, full=True)),
                   "content hash")

        metadata = metadata_root / "inference/model.py.metadata"
        lines = metadata.read_text(encoding="utf-8").splitlines()
        lines[2] = "nan"
        metadata.write_text("\n".join(lines) + "\n", encoding="utf-8")
        must_raise(SystemExit,
                   lambda: C.source_revision_evidence(snapshot, C.PINNED_REVISION),
                   "timestamp")


# ----------------------------------------------------------------- masks


def t_mask_topk():
    sc = torch.zeros(N_L, N_E)
    sc[10, :] = torch.arange(N_E).float()          # expert 255 best, 0 worst
    m = E.build_mask_from_scores(sc, KEEP, N_HASH)
    assert m["layers"]["10"]["kept"] == list(range(128, 256)), "top-k picked the wrong half"
    assert m["layers"]["10"]["pruned"] == list(range(0, 128))


def t_mask_hash_layers_untouched():
    sc = torch.rand(N_L, N_E)
    m = E.build_mask_from_scores(sc, KEEP, N_HASH)
    for lid in range(N_HASH):
        assert len(m["layers"][str(lid)]["kept"]) == N_E, f"layer {lid} was pruned"
        assert m["layers"][str(lid)]["pruned"] == []
    for lid in range(N_HASH, N_L):
        assert len(m["layers"][str(lid)]["kept"]) == KEEP


def t_mask_partition():
    """kept and pruned must partition the expert set exactly."""
    sc = torch.rand(N_L, N_E)
    m = E.build_mask_from_scores(sc, KEEP, N_HASH)
    for lid in range(N_HASH, N_L):
        k, p = m["layers"][str(lid)]["kept"], m["layers"][str(lid)]["pruned"]
        assert set(k).isdisjoint(p), f"layer {lid}: kept and pruned overlap"
        assert sorted(k + p) == list(range(N_E)), f"layer {lid}: not a partition"


def t_frequency_is_top_count():
    counts = torch.randint(0, 500, (N_L, N_E))
    m = E.build_frequency_mask(counts, KEEP, N_HASH)
    want = sorted(counts[20].argsort(descending=True)[:KEEP].tolist())
    assert m["layers"]["20"]["kept"] == want


def t_gating_is_top_total_gate_weight():
    gate_sums = torch.rand(N_L, N_E, dtype=torch.float64)
    m = E.build_gating_mask(gate_sums, KEEP, N_HASH)
    want = sorted(gate_sums[20].argsort(descending=True)[:KEEP].tolist())
    assert m["layers"]["20"]["kept"] == want


def t_random_reproducible_and_chance():
    r1 = E.build_random_mask(N_L, N_E, KEEP, N_HASH, 965)
    r2 = E.build_random_mask(N_L, N_E, KEEP, N_HASH, 965)
    r3 = E.build_random_mask(N_L, N_E, KEEP, N_HASH, 966)
    assert r1["layers"]["9"]["kept"] == r2["layers"]["9"]["kept"], "not reproducible"
    assert r1["layers"]["9"]["kept"] != r3["layers"]["9"]["kept"], "seed ignored"
    # a random half should overlap another random half at roughly chance
    ov = [len(set(r1["layers"][str(l)]["kept"]) & set(r3["layers"][str(l)]["kept"]))
          for l in range(N_HASH, N_L)]
    mean = sum(ov) / len(ov)
    assert 55 <= mean <= 73, f"random-vs-random overlap {mean:.1f} is not near chance (64)"


def t_all_variants_use_their_own_score_source():
    n_l, n_e, keep, n_hash = 5, 20, 4, 1

    def scores_for(experts, *, integer=False):
        dtype = torch.int64 if integer else torch.float32
        out = torch.zeros(n_l, n_e, dtype=dtype)
        for rank, expert in enumerate(experts):
            out[:, expert] = 100 - rank
        return out

    paper_ids, legacy_ids = [0, 1, 2, 3], [4, 5, 6, 7]
    alt_ids, frequency_ids = [8, 9, 10, 11], [12, 13, 14, 15]
    gating_ids = [16, 17, 18, 19]
    sc = scores_for(paper_ids)
    alt = scores_for(alt_ids)
    legacy = scores_for(legacy_ids)
    counts = scores_for(frequency_ids, integer=True)
    gate_sums = scores_for(gating_ids)

    v = E.all_variants(
        sc, alt, counts, gate_sums, keep, n_hash, n_l, n_e, 965, True,
        scores_mhc=sc.clone(), scores_reduced_legacy=legacy,
    )
    tags = [t for t, _ in v]
    assert tags == ["full", "pruned_paper", "pruned_reduced_legacy", "pruned_no_simibr",
                    "pruned_gating", "pruned_frequency", "pruned_random"], tags
    assert v[0][1] is None, "full variant must carry no mask"
    by_tag = dict(v)
    assert by_tag["pruned_paper"]["layers"]["2"]["kept"] == paper_ids
    assert by_tag["pruned_reduced_legacy"]["layers"]["2"]["kept"] == legacy_ids
    assert by_tag["pruned_no_simibr"]["layers"]["2"]["kept"] == alt_ids
    assert by_tag["pruned_gating"]["layers"]["2"]["kept"] == gating_ids
    assert by_tag["pruned_frequency"]["layers"]["2"]["kept"] == frequency_ids
    for tag, mask in v[1:]:
        assert len(mask["layers"]) == n_l, f"{tag} has the wrong layer count"
        assert len(mask["layers"]["2"]["kept"]) == keep, f"{tag} has the wrong keep count"
    # Without the legacy reduced-space diagnostic, that ablation is skipped.
    v2 = E.all_variants(
        sc, alt, counts, gate_sums, keep, n_hash, n_l, n_e, 965, True)
    assert "pruned_reduced_legacy" not in [t for t, _ in v2]
    # controls off
    v3 = E.all_variants(
        sc, alt, counts, gate_sums, keep, n_hash, n_l, n_e, 965, False,
        scores_mhc=sc, scores_reduced_legacy=legacy,
    )
    assert [t for t, _ in v3] == [
        "full", "pruned_paper", "pruned_reduced_legacy", "pruned_no_simibr"
    ]


def t_all_variants_reject_mismatched_tensor_shapes():
    scores = torch.rand(4, 8)
    counts = torch.ones(4, 8, dtype=torch.int64)
    gate_sums = torch.rand(4, 8)
    must_raise(
        ValueError,
        lambda: E.all_variants(
            scores, scores[:3], counts, gate_sums, 4, 1, 4, 8, 7, True),
        "shape",
    )
    must_raise(
        ValueError,
        lambda: E.all_variants(
            scores, scores, counts[:, :7], gate_sums, 4, 1, 4, 8, 7, True),
        "shape",
    )
    must_raise(
        ValueError,
        lambda: E.all_variants(
            scores, scores, counts, gate_sums, 4, 1, 5, 8, 7, True),
        "shape",
    )
    must_raise(
        ValueError,
        lambda: E.all_variants(
            scores, scores, counts, gate_sums[:, :7], 4, 1, 4, 8, 7, True),
        "shape",
    )
    different_primary = scores.clone()
    different_primary[0, 0] += 1
    must_raise(
        ValueError,
        lambda: E.all_variants(
            scores, scores, counts, gate_sums, 4, 1, 4, 8, 7, True,
            scores_mhc=different_primary,
        ),
        "alias",
    )


def t_mask_input_validation():
    sc = torch.rand(4, 8)
    for keep in (0, -1, 9, 1.5):
        must_raise(ValueError, lambda keep=keep: E.build_mask_from_scores(sc, keep, 1))
    must_raise(ValueError, lambda: E.build_mask_from_scores(torch.rand(32), 4, 1))
    must_raise(ValueError, lambda: E.build_mask_from_scores(sc, 4, -1))
    must_raise(ValueError, lambda: E.build_mask_from_scores(sc, 4, 5))
    bad = sc.clone(); bad[2, 3] = float("nan")
    must_raise(ValueError, lambda: E.build_mask_from_scores(bad, 4, 1), "nan")
    must_raise(ValueError, lambda: E.build_random_mask(4, 8, 0, 1, 7))


def t_cutoff_diagnostics_report_absolute_and_relative_margin():
    scores = torch.tensor([
        [10.0, 8.0, 6.0, 1.0],
        [4.0, 3.0, 2.7, 0.0],
    ])
    report = E.score_cutoff_diagnostics(scores, 2, 0)
    first, second = report["per_layer"]
    assert first["rank_retained"] == 2 and first["rank_pruned"] == 3
    assert abs(first["absolute_margin"] - 2.0) < 1e-12
    assert abs(first["relative_margin"] - 0.25) < 1e-12
    assert abs(second["absolute_margin"] - 0.3) < 1e-6
    assert report["minimum_margin_layer"] == 1

    no_pruning = E.score_cutoff_diagnostics(scores, 4, 0)
    assert no_pruning["minimum_relative_margin"] is None
    assert all(row["pruned_score"] is None for row in no_pruning["per_layer"])


def t_keep_must_cover_gate_topk():
    E._validate_keep(6, 256, topk=6)
    must_raise(ValueError, lambda: E._validate_keep(5, 256, topk=6), "activates")


def t_patched_gate_applies_mask_but_hash_routing_ignores_it():
    class Gate:
        def forward(self, *_args, **_kwargs):
            return "original-gate"

    class MoE:
        def forward(self, *_args, **_kwargs):
            return "original-moe"

    class Block:
        def forward(self, *_args, **_kwargs):
            return "original-block"

    official = SimpleNamespace(
        Gate=Gate, MoE=MoE, Block=Block,
        linear=lambda x, weight: F.linear(x, weight),
    )
    original_gate, original_moe, original_block = Gate.forward, MoE.forward, Block.forward
    profiler = E.Profiler(2, 4, torch.device("cpu"))
    originals = E.patch(official, profiler)
    try:
        gate = Gate()
        gate.weight = torch.eye(4)
        gate.score_func = "sigmoid"
        gate.bias = None
        gate.hash = False
        gate.topk = 2
        gate.route_scale = 1.0
        gate.ep_keep = torch.tensor([False, False, True, True])
        x = torch.tensor([[8.0, 7.0, 2.0, 1.0], [9.0, 6.0, 4.0, 3.0]])
        weights, indices = gate.forward(x, torch.tensor([0, 1]))
        assert indices.shape == weights.shape == (2, 2)
        assert set(indices.flatten().tolist()) == {2, 3}
        selected = x.sigmoid().gather(1, indices)
        want_weights = selected / selected.sum(dim=-1, keepdim=True)
        assert torch.allclose(weights, want_weights)

        hashed = Gate()
        hashed.weight = torch.eye(4)
        hashed.score_func = "sigmoid"
        hashed.bias = None
        hashed.hash = True
        hashed.topk = 2
        hashed.route_scale = 1.0
        hashed.ep_keep = torch.zeros(4, dtype=torch.bool)
        hashed.tid2eid = torch.tensor([[0, 1], [2, 3], [1, 3]])
        _, hash_indices = hashed.forward(x, torch.tensor([1, 2]))
        assert hash_indices.tolist() == [[2, 3], [1, 3]]
    finally:
        E.unpatch(official, originals)
    assert Gate.forward is original_gate
    assert MoE.forward is original_moe
    assert Block.forward is original_block


def t_patched_moe_accumulates_exact_per_token_products():
    class Gate:
        def forward(self, *_args, **_kwargs):
            raise AssertionError("class-level gate forward should not be used")

    class MoE:
        def forward(self, *_args, **_kwargs):
            return "original-moe"

    class Block:
        def forward(self, *_args, **_kwargs):
            return "original-block"

    official = SimpleNamespace(
        Gate=Gate, MoE=MoE, Block=Block,
        linear=lambda x, weight: F.linear(x, weight),
    )

    weights = torch.tensor([[0.75, 0.25], [0.40, 0.60]])
    indices = torch.tensor([[0, 1], [1, 2]])

    class FixedGate:
        hash = False

        def __call__(self, _x, _input_ids):
            return weights, indices

    class Expert:
        def __init__(self, factor):
            self.factor = factor

        def __call__(self, x, weight):
            unweighted = x * self.factor
            return unweighted if weight is None else unweighted * weight

    class Shared:
        def __call__(self, x):
            return torch.zeros_like(x)

    profiler = E.Profiler(1, 3, torch.device("cpu"))
    profiler.enabled = profiler.validate = True
    originals = E.patch(official, profiler)
    try:
        moe = MoE()
        moe.dim = 2
        moe.layer_id = 0
        moe.n_local_experts = moe.n_routed_experts = 3
        moe.experts_start_idx, moe.experts_end_idx = 0, 3
        moe.gate = FixedGate()
        moe.experts = [Expert(1.0), Expert(2.0), Expert(3.0)]
        moe.shared_experts = Shared()

        x = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
        got = moe.forward(x, torch.tensor([[10, 11]]))
        want_routed = torch.tensor([[[1.25, 0.0], [0.0, 5.2]]])
        assert got.shape == x.shape and got.dtype == x.dtype
        assert torch.allclose(got, want_routed)
        assert moe._ep_weights.shape == moe._ep_indices.shape == moe._ep_norms.shape == (2, 2)
        assert moe._ep_y_routed.shape == (2, 2)

        h = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        simibr_mhc = torch.tensor([0.2, 0.4])
        post_norm = torch.tensor([2.0, 3.0])
        moe.ep_accumulate(h, simibr_mhc, post_norm)

        expected_primary = torch.tensor([[0.3, 2.12, 4.32]], dtype=torch.float64)
        assert torch.allclose(profiler.score, expected_primary, atol=1e-12)
        assert torch.allclose(profiler.score_true, expected_primary, atol=1e-12)
        assert torch.allclose(profiler.score_mhc, expected_primary, atol=1e-12)

        reduced = (1.0 - F.cosine_similarity(h, h + want_routed.view(2, 2), dim=-1)).clamp_min(0)
        expected_legacy = torch.zeros(1, 3, dtype=torch.float64)
        expected_no_simibr = torch.zeros(1, 3, dtype=torch.float64)
        norms = torch.tensor([[1.0, 2.0], [4.0, 6.0]])
        for token in range(2):
            for slot in range(2):
                expert = int(indices[token, slot])
                expected_legacy[0, expert] += weights[token, slot] * reduced[token] * norms[token, slot]
                expected_no_simibr[0, expert] += (
                    weights[token, slot] * post_norm[token] * norms[token, slot]
                )
        assert torch.allclose(profiler.score_reduced_legacy, expected_legacy, atol=1e-12)
        assert torch.allclose(profiler.score_no_simibr, expected_no_simibr, atol=1e-12)
        assert profiler.counts.tolist() == [[1, 2, 1]]
        assert torch.allclose(profiler.gate_sums, torch.tensor([[0.75, 0.65, 0.60]]).double())
        for name in ("_ep_norms", "_ep_norms_true", "_ep_weights", "_ep_indices", "_ep_y_routed"):
            assert getattr(moe, name) is None, f"{name} was not released"
    finally:
        E.unpatch(official, originals)


def t_set_mask_sets_exact_experts_and_rejects_too_few():
    def layer(topk=2):
        gate = SimpleNamespace(ep_keep="stale", topk=topk, hash=False)
        ffn = SimpleNamespace(gate=gate, n_routed_experts=6)
        return SimpleNamespace(ffn=ffn)

    model = SimpleNamespace(layers=[layer(), layer(), layer()])
    mask = {
        "n_experts": 6,
        "layers": {
            "0": {"kept": [0, 1, 2, 3, 4, 5], "pruned": []},
            "1": {"kept": [0, 2, 4], "pruned": [1, 3, 5]},
            "2": {"kept": [1, 3, 5], "pruned": [0, 2, 4]},
        },
    }
    E.set_mask(model, mask, 1, torch.device("cpu"))
    assert model.layers[0].ffn.gate.ep_keep is None
    assert model.layers[1].ffn.gate.ep_keep.tolist() == [True, False, True, False, True, False]
    assert model.layers[2].ffn.gate.ep_keep.tolist() == [False, True, False, True, False, True]
    E.set_mask(model, None, 1, torch.device("cpu"))
    assert all(layer_.ffn.gate.ep_keep is None for layer_ in model.layers)

    too_small = json.loads(json.dumps(mask))
    too_small["layers"]["1"] = {"kept": [0], "pruned": [1, 2, 3, 4, 5]}
    must_raise(
        ValueError,
        lambda: E.set_mask(model, too_small, 1, torch.device("cpu")),
        "activates",
    )


def t_identical_scores_give_identical_masks():
    sc = torch.rand(N_L, N_E)
    a = E.build_mask_from_scores(sc, KEEP, N_HASH)
    b = E.build_mask_from_scores(sc.clone(), KEEP, N_HASH)
    assert a == b, "mask construction is not deterministic"


# ------------------------------------------------------- discrimination


def t_parse_verdict():
    cases = [("Verdict: VULNERABLE\nReasoning: x", "VULNERABLE"),
             ("Verdict: SAFE", "SAFE"),
             ("  verdict:  vulnerable (xss)", "VULNERABLE"),
             ("**Verdict:** **(SAFE).**", "SAFE"),
             ("Verdict: SAFETY UNCLEAR", None),
             ("Verdict: SAFE OR VULNERABLE", None),
             ("Verdict: VULNERABLE OR SAFE", None),
             ("Verdict: VULNERABLE or SAFE\nVerdict: SAFE", "SAFE"),
             ("Verdict: SAFE\nVerdict: SAFE", "SAFE"),
             ("Verdict: SAFE\nVerdict: VULNERABLE", None),
             ("Verdict: unclear", None),
             ("no verdict line at all", None)]
    for text, want in cases:
        got = E.parse_verdict(text)
        assert got == want, f"{text!r} -> {got}, wanted {want}"


def t_always_vulnerable_scores_chance():
    """The failure mode this metric exists to catch."""
    rows = [{"truth": t, "verdict": "VULNERABLE", "completion": "Verdict: VULNERABLE"}
            for t in ["VULNERABLE", "SAFE"] * 25]
    d = E.discrimination_stats(rows)
    assert d["tpr_recall"] == 1.0
    assert d["fpr_explicit_vulnerable"] == 1.0
    assert d["safe_abstention_rate"] == 0.0
    assert d["safe_error_rate"] == 1.0
    assert d["youden_j"] == 0.0, "a constant answerer must score J=0"
    assert d["balanced_accuracy"] == 0.5


def t_perfect_and_inverted():
    perfect = [{"truth": t, "verdict": t, "completion": "x"} for t in ["VULNERABLE", "SAFE"] * 25]
    d = E.discrimination_stats(perfect)
    assert d["youden_j"] == 1.0 and d["balanced_accuracy"] == 1.0
    flip = {"VULNERABLE": "SAFE", "SAFE": "VULNERABLE"}
    inverted = [{"truth": t, "verdict": flip[t], "completion": "x"}
                for t in ["VULNERABLE", "SAFE"] * 25]
    d2 = E.discrimination_stats(inverted)
    assert d2["youden_j"] == -1.0, "an anti-correlated model must score J=-1"


def t_unparsed_counted():
    rows = [{"truth": "VULNERABLE", "verdict": None, "completion": "x"}] * 3
    rows += [{"truth": "SAFE", "verdict": "SAFE", "completion": "x"}] * 2
    d = E.discrimination_stats(rows)
    assert d["unparsed_verdicts"] == 3
    assert d["n_vulnerable"] == 3 and d["n_safe"] == 2


def t_abstentions_cannot_score_as_correct():
    rows = [
        {"truth": "VULNERABLE", "verdict": "VULNERABLE", "completion": "x"},
        {"truth": "VULNERABLE", "verdict": None, "completion": "x"},
        {"truth": "SAFE", "verdict": "SAFE", "completion": "x"},
        {"truth": "SAFE", "verdict": None, "completion": "x"},
    ]
    d = E.discrimination_stats(rows)
    assert d["unparsed_verdicts"] == 2
    assert d["verdict_coverage"] == 0.5
    assert d["tpr_recall"] == 0.5
    assert d["tnr_specificity"] == 0.5
    assert d["fpr_explicit_vulnerable"] == 0.0
    assert d["safe_abstention_rate"] == 0.5
    assert d["safe_error_rate"] == 0.5
    assert d["tnr_specificity"] + d["safe_error_rate"] == 1.0
    assert d["youden_j"] == 0.0
    assert d["balanced_accuracy"] == 0.5
    assert d["parsed_only"]["balanced_accuracy"] == 1.0


def t_empty_discrimination_is_not_a_plausible_score():
    d = E.discrimination_stats([])
    assert d["n_vulnerable"] == 0 and d["n_safe"] == 0
    assert d["verdict_coverage"] is None
    assert d["balanced_accuracy"] is None
    assert d["youden_j"] is None


# ------------------------------------------------------------ blinding


def t_blind_hides_variant_and_covers_all():
    with tempfile.TemporaryDirectory() as td:
        src, out = Path(td) / "res", Path(td) / "judge"
        src.mkdir()
        tags = ["full", "pruned_paper", "pruned_random"]
        question_ids = [0, "Q01", "Q02", "Q03"]
        for tag in tags:
            with (src / f"answers_{tag}.jsonl").open("w") as fp:
                for i, qid in enumerate(question_ids):
                    fp.write(json.dumps({"id": qid, "cwe": f"CWE-{79+i}",
                                         "completion": f"neutral answer {i}",
                                         "variant": tag, "internal_tag": tag,
                                         "seconds": 1.0}) + "\n")

        qfile = Path(td) / "q.json"
        qfile.write_text(json.dumps([
            {"id": qid, "language": "JavaScript", "snippet": f"var x={i};",
             "vulnerability": f"issue-{i}", "cwe": f"CWE-{79+i}",
             "expected_reasoning": f"because-{i}"}
            for i, qid in enumerate(question_ids)]))

        class A:
            results, pairs, seed = str(src), False, 7
        a = A(); a.out = str(out); a.questions = str(qfile)
        E.cmd_blind(a)

        items = [json.loads(l) for l in (out / "to_judge.jsonl").read_text().splitlines()]
        assert len(items) == 12, f"expected 12 items, got {len(items)}"
        blob = json.dumps(items)
        for tag in tags:
            assert f'"{tag}"' not in blob, f"variant name {tag} leaked into the judging file"
        assert set(i["system"] for i in items) == {"A", "B", "C"}
        assert len({i["uid"] for i in items}) == 12, "uids must be unique"
        key = json.loads((out / "KEY_do_not_open_until_graded.json").read_text())
        # the published item id is opaque; the plaintext join key is withheld
        plain = key["item_to_plaintext"]
        assert set(plain) == {i["item"] for i in items}
        for it in items:
            assert plain[it["item"]] not in blob, "plaintext item key leaked"
        # the join must supply the question and a reference for every item
        for it in items:
            qid = plain[it["item"]].split("|", 1)[0]
            i = question_ids.index(0 if qid == "0" else qid)
            assert it["snippet"] == f"var x={i};", f"wrong question joined for {qid}"
            expected_prompt = E.question_text({
                "id": question_ids[i], "snippet": f"var x={i};",
                "vulnerability": f"issue-{i}", "cwe": f"CWE-{79+i}",
                "expected_reasoning": f"because-{i}", "language": "JavaScript",
            })
            assert it["prompt"] == expected_prompt
            assert it["prompt_sha256"] == hashlib.sha256(
                expected_prompt.encode()).hexdigest()
            assert it["reference"] == {
                "vulnerability": f"issue-{i}",
                "cwe": f"CWE-{79+i}",
                "expected_reasoning": f"because-{i}",
            }
            assert "variant" not in it and "internal_tag" not in it and "_variant" not in it
        assert set(key["label_to_variant"].values()) == set(tags)
        assert len(key["uid_hmac_salt_hex"]) == 64
        salt = bytes.fromhex(key["uid_hmac_salt_hex"])
        for item in items:
            variant = key["label_to_variant"][item["system"]]
            expected_uid = hmac.new(
                salt, (plain[item["item"]] + "\0" + variant).encode(),
                hashlib.sha256).hexdigest()[:16]
            assert item["uid"] == expected_uid
            assert item["uid"] != hashlib.sha1(
                (plain[item["item"]] + variant).encode()).hexdigest()[:12]
        assert ((out / "KEY_do_not_open_until_graded.json").stat().st_mode & 0o777) == 0o600
        # each system must answer every item exactly once
        from collections import Counter
        assert set(Counter(i["item"] for i in items).values()) == {3}
        systems = [i["system"] for i in items]
        transitions = sum(a != b for a, b in zip(systems, systems[1:]))
        assert transitions > 2, "records remained grouped by source variant"


def t_blind_label_mapping_is_not_derivable_from_the_seed():
    """The sealed mapping must not be a pure function of public information.

    It was: random.Random(a.seed) with --seed defaulting to 7 and the launcher
    never passing it, so shuffling ["A".."G"] and zipping against the published
    variant list recovered the whole mapping without the key.
    """
    tags = ["full", "pruned_paper", "pruned_gating", "pruned_no_simibr",
            "pruned_frequency", "pruned_random", "pruned_reduced_legacy"]
    seen = set()
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "res"
        src.mkdir()
        for tag in tags:
            with (src / f"answers_{tag}.jsonl").open("w") as fp:
                for qid in ("Q01", "Q02"):
                    fp.write(json.dumps({
                        "id": qid, "completion": "neutral", "variant": tag,
                        "snippet": "var x = 1;", "prompt": f"review {qid}",
                    }) + "\n")
        for attempt in range(6):
            out = Path(td) / f"judge{attempt}"

            class A:
                results, pairs, seed, questions = str(src), False, 7, ""
            a = A(); a.out = str(out)
            E.cmd_blind(a)
            key = json.loads(
                (out / "KEY_do_not_open_until_graded.json").read_text())
            assert set(key["label_to_variant"].values()) == set(tags)
            seen.add(tuple(sorted(key["label_to_variant"].items())))

    # A seeded shuffle gives the same mapping every time; a random one does not.
    assert len(seen) > 1, (
        "the label mapping is identical across runs at a fixed seed, so it is "
        "derivable from public information and the key file is decorative")


def t_pair_blind_includes_exact_judge_input_and_rejects_incomplete_rows():
    with tempfile.TemporaryDirectory() as td:
        src, out = Path(td) / "res", Path(td) / "judge"
        src.mkdir()
        snippet = "app.get('/x', (req, res) => res.send(req.query.q));"
        prompt = E.VERDICT_PROMPT.format(path="review_target.js", code=snippet)
        source_sha256 = hashlib.sha256(snippet.encode()).hexdigest()
        prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        prompt_ids_sha256 = E._ids_sha256([1, 2, 3])
        for tag in ("full", "pruned_paper"):
            (src / f"pairs_{tag}.jsonl").write_text(json.dumps({
                "pair_id": 4,
                "truth": "VULNERABLE",
                "path": "review_target.js",
                "snippet": snippet,
                "prompt": prompt,
                "source_sha256": source_sha256,
                "prompt_sha256": prompt_sha256,
                "prompt_ids_sha256": prompt_ids_sha256,
                "completion": "Verdict: VULNERABLE",
            }) + "\n")

        class A:
            results, pairs, seed = str(src), True, 9
            questions = ""
        a = A(); a.out = str(out)
        E.cmd_blind(a)
        items = [json.loads(line) for line in
                 (out / "to_judge.jsonl").read_text().splitlines()]
        assert len(items) == 2
        for item in items:
            assert item["snippet"] == snippet
            assert item["prompt"] == prompt
            assert item["source_sha256"] == source_sha256
            assert item["prompt_sha256"] == prompt_sha256
            assert item["prompt_ids_sha256"] == prompt_ids_sha256
            # the CodeQL label is the thing under test: it must not ship with
            # the bundle, or a grader reads the answer off the record
            assert "truth" not in item, "ground truth leaked into the judging bundle"

        key = json.loads((out / "KEY_do_not_open_until_graded.json").read_text())
        assert set(key["item_to_truth"].values()) == {"VULNERABLE"}
        assert set(key["item_to_truth"]) == {i["item"] for i in items}
        assert "VULNERABLE" not in (out / "to_judge.jsonl").read_text().replace(
            "Verdict: VULNERABLE", ""), "truth leaked in some other field"

        broken = json.loads((src / "pairs_full.jsonl").read_text())
        del broken["snippet"]
        (src / "pairs_full.jsonl").write_text(json.dumps(broken) + "\n")
        must_raise(ValueError, lambda: E.cmd_blind(a), "missing judge input")


# ------------------------------------------------ checkpoint allowlist


def t_evaluation_prompt_never_carries_the_answer():
    """The prompt must not hand the model the reference answer or the rubric.

    cmd_eval once built prompts as ``json.dumps(question)``, which put
    expected_reasoning, vulnerability and every *_terms grading list straight
    into the prompt. Both evaluation paths now share question_text.
    """
    q = {
        "id": "Q01", "language": "JavaScript", "title": "reflected xss",
        "snippet": "app.get('/hello', (req, res) => res.send(req.query.name));",
        "cwe": "CWE-79", "vulnerability": "Cross-site scripting",
        "expected_reasoning": "An attacker controls req.query.name ...",
        "accepted_terms": ["cross-site scripting", "xss", "cwe-79"],
        "sink_terms": ["res.send"], "source_terms": ["req.query"],
        "impact_terms": ["session"], "remediation_terms": ["escape"],
        "specificity_terms": ["context-aware"],
    }
    prompt = E.question_text(q)
    assert q["snippet"] in prompt, "the code under review must be in the prompt"
    leaks = ("expected_reasoning", "accepted_terms", "sink_terms", "source_terms",
             "impact_terms", "remediation_terms", "specificity_terms",
             "vulnerability")
    for key in leaks:
        assert key not in prompt, f"field name {key} leaked into the prompt"
    for value in ("An attacker controls", "Cross-site scripting", "cross-site scripting"):
        assert value not in prompt, f"answer text {value!r} leaked into the prompt"
    # and the mode that used to leak must delegate rather than build its own
    import inspect
    body = inspect.getsource(E.cmd_eval).replace(E.cmd_eval.__doc__ or "", "")
    assert "json.dumps(" not in body, \
        "cmd_eval reintroduced an independent prompt construction"
    assert "_evaluate(" in body, "cmd_eval no longer delegates to _evaluate"


def t_build_rejects_a_context_window_that_would_oom():
    """The launcher caps MAX_SEQ_LEN, but standalone modes bypass the launcher."""
    assert E.MAX_SUPPORTED_SEQ_LEN == 24576
    must_raise(ValueError,
               lambda: E.build("ckpt", "cfg", "code", 65536, 1),
               "exceeds")
    must_raise(ValueError,
               lambda: E.build("ckpt", "cfg", "code", 0, 1),
               "max_seq_len")
    # a supported window gets past the guard and fails later, on the real load
    try:
        E.build("ckpt", "cfg", "code", E.MAX_SUPPORTED_SEQ_LEN, 1)
    except ValueError as exc:
        raise AssertionError(f"supported window was rejected: {exc}")
    except Exception:
        pass


def t_no_heuristic_scoring_remains():
    """The runs must emit completions, not grades."""
    assert not hasattr(E, "score_completion"), "heuristic rubric is back"
    assert not hasattr(E, "summarize"), "heuristic aggregate is back"
    src = (HERE / "easyep_v4.py").read_text()
    for token in ("accepted_terms", "sink_terms", "remediation_terms", "specificity_terms"):
        assert token not in src, f"term-matching rubric leftover: {token}"


def t_allowed_missing_is_narrow():
    ok = ["mtp.0.embed.weight", "mtp.2.head.weight", "mtp.1.head.bias"]
    bad = ["layers.7.ffn.experts.3.w1.weight", "embed.weight", "head.weight",
           "layers.0.attn.wq_b.weight", "norm.weight", "mtp.0.ffn.gate.weight"]
    for k in ok:
        assert any(re.match(p, k) for p in E.ALLOWED_MISSING), f"{k} should be allowed"
    for k in bad:
        assert not any(re.match(p, k) for p in E.ALLOWED_MISSING), \
            f"{k} must NOT be silently allowed"


def t_checkpoint_loader_enforces_allowlist_and_unexpected_keys():
    original = E.load_model
    messages = []
    try:
        E.load_model = lambda *_args, **_kwargs: (["mtp.0.embed.weight"], [])
        result = E.load_checkpoint_checked(object(), "/tmp/model0-mp1.safetensors", messages.append)
        assert result == {"missing": ["mtp.0.embed.weight"], "unexpected": []}

        E.load_model = lambda *_args, **_kwargs: (["layers.2.ffn.experts.0.w1.weight"], [])
        must_raise(
            SystemExit,
            lambda: E.load_checkpoint_checked(object(), "/tmp/model", messages.append),
            "mismatch",
        )
        E.load_model = lambda *_args, **_kwargs: ([], ["surprise.weight"])
        must_raise(
            SystemExit,
            lambda: E.load_checkpoint_checked(object(), "/tmp/model", messages.append),
            "mismatch",
        )
    finally:
        E.load_model = original


def t_build_rejects_invalid_temperature_before_loading_model():
    for value in (-0.01, float("nan"), float("inf"), float("-inf"), True, "0.5"):
        must_raise(
            ValueError,
            lambda value=value: E.build("checkpoint", "config", "code", 128, 1, value),
            "temperature",
        )


# ------------------------------------------------------ logit compare


def t_cmp_detects_difference():
    a = [torch.randn(1, 100)]
    same = E._cmp(a, [a[0].clone()])
    assert same["max_abs"] == 0.0 and same["argmax_disagreements"] == 0
    b = a[0].clone(); b[0, int(a[0].argmax())] -= 1e4      # force a different argmax
    diff = E._cmp(a, [b])
    assert diff["max_abs"] > 1.0 and diff["argmax_disagreements"] == 1


def t_cmp_rejects_incomparable_or_nonfinite_logits():
    x = torch.randn(1, 8)
    must_raise(ValueError, lambda: E._cmp([], []), "empty")
    must_raise(ValueError, lambda: E._cmp([x, x], [x]), "length")
    must_raise(ValueError, lambda: E._cmp([x], [x.view(1, 1, 8)]), "shape")
    bad = x.clone(); bad[0, int(x.argmax())] = float("nan")
    must_raise(ValueError, lambda: E._cmp([x], [bad]), "nan")
    inf = x.clone(); inf[0, 0] = float("inf")
    must_raise(ValueError, lambda: E._cmp([inf], [x]), "infinity")


def t_parity_rejects_nonfinite_tolerances_before_loading_model():
    must_raise(
        ValueError,
        lambda: E.cmd_parity(SimpleNamespace(tol=float("nan"), floor_mult=4.0)),
        "finite",
    )
    must_raise(
        ValueError,
        lambda: E.cmd_parity(SimpleNamespace(tol=1e-3, floor_mult=float("inf"))),
        "finite",
    )


def t_global_error_tensor_includes_every_rank():
    profiler = SimpleNamespace(err_tensor=lambda: torch.tensor([0.1, 0.2]))
    assert torch.equal(
        E._global_error_tensor(profiler, 1), torch.tensor([0.1, 0.2])
    )

    original = E.dist.all_gather
    calls = 0

    def fake_all_gather(outputs, _value):
        nonlocal calls
        calls += 1
        if calls == 1:
            outputs[0].fill_(2)
            outputs[1].fill_(1)
        else:
            outputs[0].copy_(torch.tensor([0.1, 0.2]))
            outputs[1].copy_(torch.tensor([0.9, 0.0]))

    try:
        E.dist.all_gather = fake_all_gather
        got = E._global_error_tensor(profiler, 2)
    finally:
        E.dist.all_gather = original
    assert calls == 2
    assert torch.allclose(got, torch.tensor([0.1, 0.2, 0.9]))


def t_norm_error_sampling_caps_each_layer_independently():
    class Gate:
        def forward(self, *_args, **_kwargs):
            return "original-gate"

    class MoE:
        def forward(self, *_args, **_kwargs):
            return "original-moe"

    class Block:
        def forward(self, *_args, **_kwargs):
            return "original-block"

    official = SimpleNamespace(
        Gate=Gate, MoE=MoE, Block=Block,
        linear=lambda x, weight: F.linear(x, weight),
    )

    class FixedGate:
        hash = False
        topk = 1

        def __call__(self, x, _input_ids):
            return (torch.full((x.size(0), 1), 0.5),
                    torch.zeros((x.size(0), 1), dtype=torch.long))

    class Expert:
        def __call__(self, x, weight):
            unweighted = x * 2
            return unweighted if weight is None else unweighted * weight

    class Shared:
        def __call__(self, x):
            return torch.zeros_like(x)

    profiler = E.Profiler(3, 1, torch.device("cpu"))
    profiler.enabled = profiler.validate = True
    original_cap = E.MAX_ERROR_SAMPLES_PER_LAYER_PER_RANK
    originals = E.patch(official, profiler)
    try:
        E.MAX_ERROR_SAMPLES_PER_LAYER_PER_RANK = 2
        for layer_id in (0, 0, 1):
            moe = MoE()
            moe.dim = 2
            moe.layer_id = layer_id
            moe.n_local_experts = moe.n_routed_experts = 1
            moe.experts_start_idx, moe.experts_end_idx = 0, 1
            moe.gate = FixedGate()
            moe.experts = [Expert()]
            moe.shared_experts = Shared()
            moe.forward(torch.ones(1, 3, 2), torch.arange(3).view(1, 3))
            moe.ep_release()
    finally:
        E.MAX_ERROR_SAMPLES_PER_LAYER_PER_RANK = original_cap
        E.unpatch(official, originals)

    assert profiler._err_n_by_layer == [2, 2, 0]
    assert profiler.err_tensor().numel() == 4
    assert E._global_error_sample_counts(profiler, 1) == [2, 2, 0]


def t_quantiles_are_float32_default_dtype_safe():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        err = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
        got = E._quantiles(err, (0.5, 1.0))
        assert got == [1.5, 3.0]
        empty = E._quantiles(torch.empty(0, dtype=torch.float32), (0.5, 1.0))
        assert all(value != value for value in empty), "empty quantiles must be NaN"
        bad = err.clone(); bad[0] = float("nan")
        must_raise(ValueError, lambda: E._quantiles(bad), "nan")
    finally:
        torch.set_default_dtype(previous)


# -------------------------------------------------- hc_post inlining


def t_mhc_residual_base_matches_hc_post():
    """The production helper must compute C @ residual, not a parallel formula here."""
    torch.manual_seed(0)
    b, s, hc, d = 2, 5, 4, 16
    residual, post = torch.randn(b, s, hc, d), torch.randn(b, s, hc)
    comb, y_r = torch.randn(b, s, hc, hc), torch.randn(b, s, d)

    def hc_post(x):                                    # verbatim from V4 model.py
        y = (post.unsqueeze(-1) * x.unsqueeze(-2)
             + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2))
        return y.type_as(x)

    base = E.mhc_residual_base(residual, comb)
    routed = post.unsqueeze(-1) * y_r.unsqueeze(-2) + base
    assert torch.allclose(base, hc_post(torch.zeros_like(y_r)), atol=1e-6)
    assert torch.allclose(routed, hc_post(y_r), atol=1e-6)
    sim = (1 - F.cosine_similarity(base.flatten(2), routed.flatten(2), dim=-1)).clamp_min(0)
    assert sim.view(-1).numel() == b * s, "simibr_mhc must be one value per token"
    assert (sim >= 0).all(), "clamp_min(0) violated"


def t_mhc_residual_base_is_stable_for_bfloat16():
    """Do not recover a small residual term by subtracting rounded large tensors."""
    torch.manual_seed(1)
    b, s, hc, d = 1, 3, 4, 8
    residual = torch.randn(b, s, hc, d).bfloat16()
    comb = torch.randn(b, s, hc, hc).bfloat16()
    want = torch.sum(
        comb.float().unsqueeze(-1) * residual.float().unsqueeze(-2), dim=2
    )
    got = E.mhc_residual_base(residual, comb)
    assert got.dtype == torch.float32, "mHC baseline must be accumulated in float32"
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-6)

    post = torch.ones(b, s, hc, dtype=torch.bfloat16)
    x_ffn = torch.full((b, s, d), 8192.0, dtype=torch.bfloat16)
    rounded_post = (post.unsqueeze(-1) * x_ffn.unsqueeze(-2)).bfloat16()
    rounded_full = (rounded_post.float() + want).bfloat16()
    subtractive = rounded_full.float() - rounded_post.float()
    assert not torch.allclose(subtractive, want, atol=1e-2, rtol=1e-2), \
        "fixture no longer exposes BF16 cancellation"


def t_patched_block_uses_real_mhc_baseline_and_post_norm():
    class Gate:
        def forward(self, *_args, **_kwargs):
            return "original-gate"

    class MoE:
        def forward(self, *_args, **_kwargs):
            return "original-moe"

    class Block:
        def forward(self, *_args, **_kwargs):
            return "original-block"

    official = SimpleNamespace(
        Gate=Gate, MoE=MoE, Block=Block,
        linear=lambda x, weight: F.linear(x, weight),
    )
    profiler = E.Profiler(1, 3, torch.device("cpu"))
    profiler.enabled = True

    post = torch.tensor([[[0.5, 1.5], [2.0, 0.25]]])
    comb = torch.tensor([[[[0.8, 0.2], [0.1, 0.9]],
                          [[0.3, 0.7], [0.6, 0.4]]]])
    y_routed = torch.tensor([[[0.4, -0.6, 0.2], [-0.3, 0.5, 0.7]]])
    shared = torch.tensor([[[0.1, 0.2, -0.2], [0.2, -0.1, 0.3]]])

    class ZeroAttention:
        def __call__(self, x, _start_pos, *_args):
            return torch.zeros_like(x)

    class CaptureFFN:
        def __init__(self):
            self.gate = SimpleNamespace(hash=False)
            self._ep_norms = None
            self._ep_y_routed = None
            self.accumulate_args = None

        def __call__(self, x, _input_ids):
            assert x.shape == y_routed.shape
            self._ep_norms = torch.ones(x.numel() // x.size(-1), 1)
            self._ep_y_routed = y_routed.reshape(-1, y_routed.size(-1))
            return y_routed + shared

        def ep_accumulate(self, *args):
            self.accumulate_args = args

    block = Block()
    block.layer_id = 0
    block.hc_attn_fn = block.hc_ffn_fn = None
    block.hc_attn_scale = block.hc_ffn_scale = 1.0
    block.hc_attn_base = block.hc_ffn_base = 0.0
    block.attn_norm = block.ffn_norm = lambda x: x
    block.attn = ZeroAttention()
    block.ffn = CaptureFFN()

    def hc_pre(x, *_args):
        return x.float().mean(dim=2).type_as(x), post.type_as(x), comb.type_as(x)

    def hc_post(value, residual, post_, comb_):
        base = E.mhc_residual_base(residual, comb_)
        return (base + post_.float().unsqueeze(-1) * value.float().unsqueeze(-2)).type_as(value)

    block.hc_pre, block.hc_post = hc_pre, hc_post
    initial = torch.tensor([[[[1.0, -0.2, 0.5], [0.3, 0.7, -0.4]],
                             [[-0.5, 0.4, 1.2], [0.8, -0.6, 0.1]]]])

    originals = E.patch(official, profiler)
    try:
        got = block.forward(initial, 0, torch.tensor([[4, 5]]))
    finally:
        E.unpatch(official, originals)

    first_residual = E.mhc_residual_base(initial, comb)
    expected_h = first_residual.mean(dim=2)
    base = E.mhc_residual_base(first_residual, comb)
    routed = base + post.unsqueeze(-1) * y_routed.unsqueeze(-2)
    expected_sim = (1.0 - F.cosine_similarity(
        base.flatten(2), routed.flatten(2), dim=-1
    )).clamp_min(0).reshape(-1)
    expected_post_norm = post.float().norm(dim=-1).reshape(-1)
    expected_output = base + post.unsqueeze(-1) * (y_routed + shared).unsqueeze(-2)

    assert torch.allclose(got, expected_output, atol=1e-6)
    assert block.ffn.accumulate_args is not None
    h_flat, simibr_mhc, post_norm = block.ffn.accumulate_args
    assert h_flat.shape == (2, 3)
    assert simibr_mhc.shape == post_norm.shape == (2,)
    assert torch.allclose(h_flat, expected_h.reshape(2, 3), atol=1e-6)
    assert torch.allclose(simibr_mhc, expected_sim, atol=1e-6)
    assert torch.allclose(post_norm, expected_post_norm, atol=1e-6)


# ------------------------------------------------- profiler bookkeeping


def t_profiler_shapes_and_dtypes():
    p = E.Profiler(N_L, N_E, torch.device("cpu"))
    for name in ("score", "score_mhc", "score_reduced_legacy", "score_no_simibr",
                 "score_true", "gate_sums"):
        t = getattr(p, name)
        assert t.shape == (N_L, N_E), f"{name} shape {tuple(t.shape)}"
        assert t.dtype == torch.float64, f"{name} must be float64 to sum many small products"
    assert p.counts.dtype == torch.int64
    assert p.enabled is False and p.validate is False, "profiling must default to off"
    st = p.state()
    for k in ("score", "score_mhc", "score_reduced_legacy", "score_no_simibr",
              "score_true", "counts", "gate_sums", "norm_rel_err", "tokens_seen",
              "n_layers", "n_experts", "accumulation_mode",
              "deterministic_score_reductions", "norm_error_samples_per_layer"):
        assert k in st, f"state() is missing {k}"
    assert st["accumulation_mode"] == E.ACCUMULATION_MODE
    # The attestation is an observation, not a constant: a profiler that never
    # accumulated has observed nothing and must not claim the guarantee.
    assert st["deterministic_score_reductions"] is False
    p.note_reduction_determinism(True)
    assert p.state()["deterministic_score_reductions"] is True
    # one non-deterministic reduction withdraws the claim for the whole run
    p.note_reduction_determinism(False)
    assert p.state()["deterministic_score_reductions"] is False
    assert st["norm_error_samples_per_layer"] == [0] * N_L


def t_deterministic_score_reduction_scope_restores_global_setting():
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        target = torch.zeros(2, dtype=torch.float64)
        with E.deterministic_score_reductions():
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            target.index_add_(0, torch.tensor([0, 0, 1]),
                              torch.tensor([0.25, 0.5, 1.0], dtype=torch.float64))
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.equal(target, torch.tensor([0.75, 1.0], dtype=torch.float64))
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)


def t_score_artifact_schema_rejects_stale_or_incomplete_scores():
    identity = fixture_model_identity()
    profiler = E.Profiler(4, 8, torch.device("cpu"))
    profiler.gate_topk = 2
    profiler.tokens_seen = 1
    profiler.note_reduction_determinism(True)   # as a real accumulation would
    state = profiler.state(1, identity)
    state["calibration"] = fixture_calibration_provenance()
    assert E._require_score_artifact(
        state, "fixture", n_hash_layers=1, model_identity=identity) is state

    stale = dict(state); stale["score_schema_version"] -= 1
    must_raise(SystemExit, lambda: E._require_score_artifact(stale, "fixture"), "re-profile")
    wrong = dict(state); wrong["score_semantics"] = "old-reduced-space-score"
    must_raise(SystemExit, lambda: E._require_score_artifact(wrong, "fixture"), "re-profile")
    missing = dict(state); del missing["counts"]
    must_raise(SystemExit, lambda: E._require_score_artifact(missing, "fixture"), "missing")
    # an artifact whose reductions were observed non-deterministic is rejected,
    # not silently ranked
    withdrawn = dict(state); withdrawn["deterministic_score_reductions"] = False
    must_raise(
        SystemExit,
        lambda: E._require_score_artifact(withdrawn, "fixture"),
        "deterministic")

    nondeterministic = dict(state); del nondeterministic["accumulation_mode"]
    must_raise(
        SystemExit,
        lambda: E._require_score_artifact(nondeterministic, "fixture"),
        "deterministic",
    )
    must_raise(
        SystemExit,
        lambda: E._require_score_artifact(state, "fixture", n_hash_layers=2),
        "n_hash_layers",
    )
    bare = E.Profiler(4, 8, torch.device("cpu"))
    bare.note_reduction_determinism(True)   # isolate the provenance failure
    no_provenance = bare.state()
    must_raise(
        SystemExit, lambda: E._require_score_artifact(no_provenance, "fixture"),
        "provenance",
    )
    no_calibration = dict(state); del no_calibration["calibration"]
    must_raise(
        SystemExit, lambda: E._require_score_artifact(no_calibration, "fixture"),
        "calibration provenance",
    )
    other_identity = fixture_model_identity("1")
    must_raise(
        SystemExit,
        lambda: E._require_score_artifact(
            state, "fixture", n_hash_layers=1, model_identity=other_identity),
        "model/runtime",
    )
    tampered = dict(state)
    tampered["model_identity"] = other_identity
    must_raise(
        SystemExit, lambda: E._require_score_artifact(tampered, "fixture"),
        "digest",
    )


def t_model_identity_hashes_sources_tokenizer_shard_and_optional_attestation():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        code = root / "inference"
        encoding = root / "encoding"
        checkpoint = root / "checkpoint"
        code.mkdir(); encoding.mkdir(); checkpoint.mkdir()
        (code / "model.py").write_text("MODEL = 1\n")
        (code / "generate.py").write_text("def generate(): pass\n")
        (code / "kernel.py").write_text("KERNEL = 1\n")
        (code / "convert.py").write_text("CONVERTER = 1\n")
        (encoding / "encoding_dsv4.py").write_text("ENCODING = 1\n")
        config = root / "config.json"
        config.write_text('{"n_layers": 4}\n')
        (code / "config.json").write_bytes(config.read_bytes())
        (checkpoint / "tokenizer.json").write_text('{"version": "1"}\n')
        (checkpoint / "tokenizer_config.json").write_text('{"eos": 0}\n')
        shard = checkpoint / "model0-mp1.safetensors"
        shard.write_bytes(bytes(range(256)) * 1024)

        first = E._model_identity(str(checkpoint), str(config), str(code), 1)
        second = E._model_identity(str(checkpoint), str(config), str(code), 1)
        assert first == second, "unchanged model inputs must have a stable identity"
        assert first["config"]["sha256"] == E._sha256_file(config)
        assert set(first["tokenizer"]) == {"tokenizer.json", "tokenizer_config.json"}
        assert "inference/kernel.py" in first["official_sources"]
        assert len(first["checkpoint"]["shards"][0]["sample_sha256"]) == 64

        (code / "model.py").write_text("MODEL = 2\n")
        changed = E._model_identity(str(checkpoint), str(config), str(code), 1)
        assert changed != first
        assert "official_sources" in E._identity_difference(first, changed)

        sidecar = checkpoint / "EASYEP_CHECKPOINT_PROVENANCE.json"
        sidecar.write_text(json.dumps({"content_identity_sha256": "f" * 64}) + "\n")
        must_raise(
            (ValueError, SystemExit),
            lambda: E._model_identity(str(checkpoint), str(config), str(code), 1),
            "sidecar",
        )
        stat = shard.stat()
        source_hashes = {
            relative: C.sha256_file(root / relative)
            for relative in C.REQUIRED_SOURCE_FILES
        }
        provenance = {
            "schema_version": C.SCHEMA_VERSION,
            "model": {"id": C.PINNED_MODEL_ID, "revision": C.PINNED_REVISION,
                      "inference_revision": C.PINNED_REVISION},
            "conversion": {
                "source_files_sha256": source_hashes,
                "n_experts": 256, "model_parallel": 1, "expert_dtype": "fp4",
            },
            "checkpoint": {
                "nproc": 1,
                "shards": [{"name": shard.name, "size": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns,
                            "sha256": C.sha256_file(shard)}],
            },
        }
        provenance["content_identity_sha256"] = C.checkpoint_content_identity(provenance)
        sidecar.write_text(json.dumps(provenance) + "\n")
        attested = E._model_identity(str(checkpoint), str(config), str(code), 1)
        record = attested["checkpoint"]["full_hash_attestation"]
        assert record["name"] == sidecar.name
        assert record["content_identity_sha256"] == provenance["content_identity_sha256"]


def t_calibration_chunking_covers_the_whole_file():
    """Truncation drops late sinks; token-exact chunking must not.

    Exercises the path the profiler actually runs (_calibration_chunks ->
    split_to_fit), measured in tokens, not a char-budget helper.
    """
    body = "".join(f"line {i} some code here\n" for i in range(400))

    class Tokenizer:
        eos_token_id = 0

        def encode(self, prompt):
            return [ord(c) for c in prompt]

    def encode_messages(messages, thinking_mode):
        assert messages and thinking_mode == "chat"
        return messages[0]["content"]

    def pieces_of(inputs):
        """Recover each chunk's source text from its prompt tokens."""
        out = []
        for item in inputs:
            prompt = "".join(chr(i) for i in item["prompt_ids"])
            opener = "<UNTRUSTED_CODE>\n"
            begin = prompt.index(opener) + len(opener)
            out.append(prompt[begin:prompt.rindex("\n</UNTRUSTED_CODE>")])
        return out

    def chunk(code, **kw):
        return E._calibration_chunks(
            [("CWE-1/a.js", code)], Tokenizer(), encode_messages,
            max_seq_len=1000, max_new_tokens=8, **kw)

    inputs = chunk(body, max_chunks=0)  # zero means unlimited
    assert len(inputs) > 1, "long file was not chunked"
    assert all(len(i["prompt_ids"]) <= 1000 - 8 for i in inputs), "chunk exceeded budget"
    assert "".join(pieces_of(inputs)) == body, "chunking lost or reordered content"
    assert all(i["n_chunks"] == len(inputs) for i in inputs), "n_chunks disagrees"
    assert [i["chunk_index"] for i in inputs] == list(range(len(inputs)))

    # short files pass through untouched
    assert pieces_of(chunk("short enough to fit\n", max_chunks=0)) == ["short enough to fit\n"]

    # An explicit cap must reject overflow instead of silently dropping the tail.
    must_raise(ValueError, lambda: chunk(body, max_chunks=2), "chunk")


def t_calibration_sampling_is_stratified_and_stable():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for cwe in ("CWE-79", "CWE-89", "CWE-22"):
            (root / cwe).mkdir(parents=True)
            for i in range(10):
                (root / cwe / f"f{i}.js").write_text(f"// {cwe} {i}\nvar x = {i};\n")
        a = E.sample_calibration_files(root, 9)
        b = E.sample_calibration_files(root, 9)
        assert len(a) == 9, f"requested 9 samples, got {len(a)}"
        assert [p for p, _ in a] == [p for p, _ in b], "sampling must be reproducible"
        for path, content in a:
            assert content == (root / path).read_text(), f"content/path mismatch for {path}"
        cwes = {p.split("/")[0] for p, _ in a}
        assert cwes == {"CWE-79", "CWE-89", "CWE-22"}, f"not stratified: {cwes}"
        per = {c: sum(1 for p, _ in a if p.startswith(c)) for c in cwes}
        assert max(per.values()) - min(per.values()) <= 1, f"unbalanced: {per}"


def t_calibration_sampling_randomizes_strata_when_n_is_small():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        strata = {f"CWE-{i:03d}" for i in range(12)}
        for cwe in sorted(strata):
            (root / cwe).mkdir()
            (root / cwe / "sample.js").write_text(f"// {cwe}\n")

        selections = []
        for seed in range(12):
            got = E.sample_calibration_files(root, 4, seed=seed)
            selected = {p.split("/")[0] for p, _ in got}
            assert len(got) == len(selected) == 4
            assert selected <= strata
            selections.append(selected)
        assert len({frozenset(x) for x in selections}) > 1, "seed does not affect stratum choice"
        assert set().union(*selections) == strata, "some strata are never eligible when n < strata"


def t_calibration_sampling_rejects_empty_or_insufficient_inputs():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        must_raise(ValueError, lambda: E.sample_calibration_files(root, 1), "eligible")
        (root / "CWE-001").mkdir()
        (root / "CWE-001" / "one.js").write_text("let x = 1;\n")
        must_raise(ValueError, lambda: E.sample_calibration_files(root, 0), "positive")
        must_raise(ValueError, lambda: E.sample_calibration_files(root, 2), "only 1")
        must_raise(ValueError, lambda: E.sample_calibration_files(root / "missing", 1), "directory")


def t_calibration_profiles_prompt_and_generated_response_once():
    class Tokenizer:
        eos_token_id = 0

        def encode(self, _prompt):
            return list(range(40))

    def encode_messages(messages, thinking_mode):
        assert messages and thinking_mode == "chat"
        return "encoded"

    inputs = E._calibration_chunks(
        [("CWE-1/a.js", "const x = 1;\n")], Tokenizer(), encode_messages,
        max_seq_len=64, max_new_tokens=8, max_chunks=0,
    )
    assert len(inputs) == 1 and inputs[0]["prompt_ids"] == list(range(40))

    profiler = SimpleNamespace(enabled=False, tokens_seen=0)
    seen = []

    class Model:
        def forward(self, tokens, start_pos):
            assert profiler.enabled is True and start_pos == 0
            seen.append(tokens.squeeze(0).tolist())
            profiler.tokens_seen += tokens.numel()

    def generate(_model, prompts, max_new_tokens, eos_token_id):
        assert profiler.enabled is False
        assert max_new_tokens == 8 and eos_token_id == 0
        return [prompts[0] + [50, 51, 52]]  # prompt-inclusive API form

    result = E._run_calibration(
        Model(), inputs, generate, 0, profiler, 64, 8, 123,
        torch.device("cpu"),
    )
    assert result["forwards"] == 1 and result["response_tokens"] == 3
    assert result["generated_responses"][0]["tokens"] == 3
    assert seen == [list(range(40)) + [50, 51, 52]]
    assert profiler.enabled is False


def t_calibration_drops_only_synthetic_budget_overflow_eos():
    prompt_ids = list(range(10))
    inputs = [{"prompt_ids": prompt_ids, "path": "x.js", "file_index": 0,
               "chunk_index": 0, "n_chunks": 1}]
    profiler = SimpleNamespace(enabled=False, tokens_seen=0)
    seen = []

    class Model:
        def forward(self, tokens, start_pos):
            assert profiler.enabled and start_pos == 0
            seen.append(tokens.squeeze(0).tolist())
            profiler.tokens_seen += tokens.numel()

    real_completion = [50, 51, 52, 53, 54]

    def generate(_model, prompts, max_new_tokens, eos_token_id):
        assert max_new_tokens == len(real_completion) and eos_token_id == 0
        # The pinned generator is completion-only. Deliberately exercise the
        # helper's alternate prompt-inclusive compatibility form here; the
        # trailing EOS models the pinned generator's budget-overflow sentinel.
        return [prompts[0] + real_completion + [eos_token_id]]

    result = E._run_calibration(
        Model(), inputs, generate, 0, profiler, 32, len(real_completion), 7,
        torch.device("cpu"),
    )
    assert result["forwards"] == 1
    assert result["response_tokens"] == len(real_completion)
    assert result["generated_responses"][0]["token_ids_sha256"] == E._ids_sha256(real_completion)
    assert seen == [prompt_ids + real_completion], "a real generated token was dropped"
    assert profiler.enabled is False


def t_calibration_rejects_no_usable_inputs_or_response():
    profiler = SimpleNamespace(enabled=False, tokens_seen=0)
    must_raise(
        ValueError,
        lambda: E._run_calibration(
            object(), [], lambda *_args: [[1]], 0, profiler, 64, 8, 1,
            torch.device("cpu"),
        ),
        "calibration",
    )

    item = {"prompt_ids": list(range(40)), "path": "x.js",
            "file_index": 0, "chunk_index": 0, "n_chunks": 1}
    must_raise(
        RuntimeError,
        lambda: E._run_calibration(
            object(), [item], lambda *_args: [[]], 0, profiler, 64, 8, 1,
            torch.device("cpu"),
        ),
        "no response",
    )
    assert profiler.enabled is False


def t_calibration_provenance_binds_inputs_responses_and_parameters():
    files = [("CWE-001/sample.js", "const value = 1;\n")]
    inputs = [{"path": files[0][0], "file_index": 0, "chunk_index": 0,
               "n_chunks": 1, "prompt_ids": [4, 5, 6]}]
    result = {"forwards": 1, "response_tokens": 2,
              "generated_responses": [{"run_index": 0, "path": files[0][0],
                                       "file_index": 0, "chunk_index": 0,
                                       "tokens": 2,
                                       "token_ids_sha256": E._ids_sha256([7, 8])}]}
    record = E._calibration_provenance(
        files, inputs, result, seed=965, temperature=0.6,
        max_seq_len=64, max_new_tokens=8, max_chunks=0,
        source_kind="corpus_files", security_prompt=True)
    E._validate_calibration_provenance(record)
    assert record["selected_sources"][0]["text_sha256"] == \
        hashlib.sha256(files[0][1].encode()).hexdigest()
    assert record["prompt_chunks"][0]["prompt_ids_sha256"] == E._ids_sha256([4, 5, 6])

    tampered = json.loads(json.dumps(record))
    tampered["parameters"]["temperature"] = float("nan")
    must_raise(SystemExit, lambda: E._validate_calibration_provenance(tampered),
               "temperature")


def main():
    print("easyep_v4 logic tests\n")
    for name, fn in sorted(globals().items()):
        if name.startswith("t_") and callable(fn):
            check(name[2:], fn)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for n, why in FAIL:
            print(f"  {n}: {why}")
        sys.exit(1)


if __name__ == "__main__":
    main()
