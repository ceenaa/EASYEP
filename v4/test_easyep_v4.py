"""Tests for the GPU-free logic in easyep_v4.py.

Everything here runs on a login node in seconds. It covers the parts a reviewer
would want to check by reading -- mask construction, the control baselines, the
discrimination metric, blinding -- so those can be verified without a 164 GiB
model load.

    python test_easyep_v4.py
"""
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("e", HERE / "easyep_v4.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)

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


N_L, N_E, KEEP, N_HASH = 43, 256, 128, 3


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


def t_all_variants_shape():
    sc, alt, mhc = torch.rand(N_L, N_E), torch.rand(N_L, N_E), torch.rand(N_L, N_E)
    counts = torch.randint(0, 100, (N_L, N_E))
    v = E.all_variants(sc, alt, counts, KEEP, N_HASH, N_L, N_E, 965, True, mhc)
    tags = [t for t, _ in v]
    assert tags == ["full", "pruned_paper", "pruned_mhc", "pruned_no_simibr",
                    "pruned_frequency", "pruned_random"], tags
    assert v[0][1] is None, "full variant must carry no mask"
    # without mHC scores the variant is skipped, not faked
    v2 = E.all_variants(sc, alt, counts, KEEP, N_HASH, N_L, N_E, 965, True, None)
    assert "pruned_mhc" not in [t for t, _ in v2]
    # controls off
    v3 = E.all_variants(sc, alt, counts, KEEP, N_HASH, N_L, N_E, 965, False, mhc)
    assert [t for t, _ in v3] == ["full", "pruned_paper", "pruned_mhc", "pruned_no_simibr"]


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
    assert d["fpr_false_alarm"] == 1.0
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


# --------------------------------------------------------- term rubric


def t_score_completion_recall_only_is_gameable():
    """Documents the weakness that motivated the paired eval: a wall of
    security vocabulary scores well on recall alone."""
    q = {"cwe": "CWE-79", "accepted_terms": ["cross-site scripting", "xss", "cwe-79"],
         "sink_terms": ["res.send"], "source_terms": ["req.query.name"],
         "remediation_terms": ["escape"]}
    keyword_soup = ("Verdict: VULNERABLE. cross-site scripting xss cwe-79 res.send "
                    "req.query.name escape sql injection path traversal")
    s = E.score_completion(q, keyword_soup)
    assert s["accepted"] == 1.0 and s["verdict_vulnerable"] is True
    assert s["cwe_mentioned"] is True


def t_score_completion_fields():
    q = {"cwe": "CWE-89", "accepted_terms": ["sql injection"], "sink_terms": ["query("],
         "source_terms": [], "remediation_terms": ["parameterized"]}
    s = E.score_completion(q, "Verdict: SAFE\nNothing here.")
    assert s["verdict"] == "SAFE" and s["verdict_vulnerable"] is False
    assert s["accepted"] == 0.0
    assert s["source"] is None, "an empty term list must yield None, not 0.0"


def t_summarize_ignores_none():
    rows = [{"seconds": 1.0, "score": {"accepted": 1.0, "sink": None, "source": None,
                                       "remediation": 0.5, "verdict_vulnerable": True,
                                       "cwe_mentioned": True}},
            {"seconds": 3.0, "score": {"accepted": 0.0, "sink": None, "source": None,
                                       "remediation": 0.5, "verdict_vulnerable": False,
                                       "cwe_mentioned": False}}]
    out = E.summarize(rows)
    assert out["accepted"] == 0.5 and out["remediation"] == 0.5
    assert out["sink"] is None
    assert out["verdict_vulnerable"] == 0.5 and out["mean_seconds"] == 2.0


# ------------------------------------------------------------ blinding


def t_blind_hides_variant_and_covers_all():
    with tempfile.TemporaryDirectory() as td:
        src, out = Path(td) / "res", Path(td) / "judge"
        src.mkdir()
        tags = ["full", "pruned_paper", "pruned_random"]
        for tag in tags:
            with (src / f"answers_{tag}.jsonl").open("w") as fp:
                for i in range(4):
                    fp.write(json.dumps({"id": f"Q{i:02d}", "cwe": "CWE-79",
                                         "completion": f"{tag} answer {i}",
                                         "seconds": 1.0,
                                         "score": {"accepted": 0.0}}) + "\n")

        class A:
            results, out_, pairs, seed = str(src), str(out), False, 7
        a = A(); a.out = str(out)
        E.cmd_blind(a)

        items = [json.loads(l) for l in (out / "to_judge.jsonl").read_text().splitlines()]
        assert len(items) == 12, f"expected 12 items, got {len(items)}"
        blob = json.dumps(items)
        for tag in tags:
            assert f'"{tag}"' not in blob, f"variant name {tag} leaked into the judging file"
        assert set(i["system"] for i in items) == {"A", "B", "C"}
        assert len({i["uid"] for i in items}) == 12, "uids must be unique"
        key = json.loads((out / "KEY_do_not_open_until_graded.json").read_text())
        assert set(key["label_to_variant"].values()) == set(tags)
        # each system must answer every item exactly once
        from collections import Counter
        assert set(Counter(i["item"] for i in items).values()) == {3}


# ------------------------------------------------ checkpoint allowlist


def t_allowed_missing_is_narrow():
    ok = ["mtp.0.embed.weight", "mtp.2.head.weight", "mtp.1.head.bias"]
    bad = ["layers.7.ffn.experts.3.w1.weight", "embed.weight", "head.weight",
           "layers.0.attn.wq_b.weight", "norm.weight", "mtp.0.ffn.gate.weight"]
    for k in ok:
        assert any(re.match(p, k) for p in E.ALLOWED_MISSING), f"{k} should be allowed"
    for k in bad:
        assert not any(re.match(p, k) for p in E.ALLOWED_MISSING), \
            f"{k} must NOT be silently allowed"


# ------------------------------------------------------ logit compare


def t_cmp_detects_difference():
    a = [torch.randn(1, 100)]
    same = E._cmp(a, [a[0].clone()])
    assert same["max_abs"] == 0.0 and same["argmax_disagreements"] == 0
    b = a[0].clone(); b[0, int(a[0].argmax())] -= 1e4      # force a different argmax
    diff = E._cmp(a, [b])
    assert diff["max_abs"] > 1.0 and diff["argmax_disagreements"] == 1


# -------------------------------------------------- hc_post inlining


def t_inlined_hc_post_matches():
    """The mHC simibr inlines hc_post(0, ...) as the residual-mixing term."""
    torch.manual_seed(0)
    b, s, hc, d = 2, 5, 4, 16
    residual, post = torch.randn(b, s, hc, d), torch.randn(b, s, hc)
    comb, y_r = torch.randn(b, s, hc, hc), torch.randn(b, s, d)

    def hc_post(x):                                    # verbatim from V4 model.py
        y = (post.unsqueeze(-1) * x.unsqueeze(-2)
             + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2))
        return y.type_as(x)

    base = torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    routed = post.unsqueeze(-1) * y_r.unsqueeze(-2) + base
    assert torch.allclose(base, hc_post(torch.zeros_like(y_r)), atol=1e-6)
    assert torch.allclose(routed, hc_post(y_r), atol=1e-6)
    sim = (1 - F.cosine_similarity(base.flatten(2), routed.flatten(2), dim=-1)).clamp_min(0)
    assert sim.view(-1).numel() == b * s, "simibr_mhc must be one value per token"
    assert (sim >= 0).all(), "clamp_min(0) violated"


# ------------------------------------------------- profiler bookkeeping


def t_profiler_shapes_and_dtypes():
    p = E.Profiler(N_L, N_E, torch.device("cpu"))
    for name in ("score", "score_mhc", "score_no_simibr", "score_true", "gate_sums"):
        t = getattr(p, name)
        assert t.shape == (N_L, N_E), f"{name} shape {tuple(t.shape)}"
        assert t.dtype == torch.float64, f"{name} must be float64 to sum many small products"
    assert p.counts.dtype == torch.int64
    assert p.enabled is False and p.validate is False, "profiling must default to off"
    st = p.state()
    for k in ("score", "score_mhc", "score_no_simibr", "score_true", "counts",
              "gate_sums", "norm_rel_err", "tokens_seen", "n_layers", "n_experts"):
        assert k in st, f"state() is missing {k}"


def t_calibration_sampling_is_stratified_and_stable():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for cwe in ("CWE-79", "CWE-89", "CWE-22"):
            (root / cwe).mkdir(parents=True)
            for i in range(10):
                (root / cwe / f"f{i}.js").write_text(f"// {cwe} {i}\nvar x = {i};\n")
        a = E.sample_calibration_files(root, 9)
        b = E.sample_calibration_files(root, 9)
        assert [p for p, _ in a] == [p for p, _ in b], "sampling must be reproducible"
        cwes = {p.split("/")[0] for p, _ in a}
        assert cwes == {"CWE-79", "CWE-89", "CWE-22"}, f"not stratified: {cwes}"
        per = {c: sum(1 for p, _ in a if p.startswith(c)) for c in cwes}
        assert max(per.values()) - min(per.values()) <= 1, f"unbalanced: {per}"


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
