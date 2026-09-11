import json
from types import SimpleNamespace

import pytest

from freetoken.pruning.artifacts import checkpoint_identity, read_json, write_new
from freetoken.pruning import workflow as w
from test_artifacts import checkpoint


def manifest(tmp_path, rows):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    return path


def test_source_is_read_not_executed_and_labels_stay_out_of_prompt(tmp_path, checkpoint, monkeypatch):
    (tmp_path / "CWE_001_vulnerable.py").write_text('raise RuntimeError("must never execute")')
    (tmp_path / "fixed_safe.py").write_text('x = 1')
    rows = [{"id": "one_vulnerable", "repository": "repo_CWE", "split": "calibration",
             "files": ["CWE_001_vulnerable.py", "fixed_safe.py"],
             "expected_findings": [{"id": "secret-label"}]}]
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text.encode()))
    manager = SimpleNamespace(render_prompt=lambda msg: msg.text[0]["content"])
    monkeypatch.setattr(w, "tokenizer_tools", lambda path: (tokenizer, manager))
    output = tmp_path / "prepared.json"
    args = SimpleNamespace(model_dir=checkpoint, manifest=manifest(tmp_path, rows), source_root=tmp_path,
        split="calibration", max_bytes=1000, max_prompt_tokens=1000, thinking_mode="thinking", output=output)
    w.prepare(args)
    case = read_json(output)["cases"][0]
    assert "must never execute" in case["raw_prompt"]
    assert "secret-label" not in case["raw_prompt"]
    assert all(word not in case["raw_prompt"] for word in ("CWE", "vulnerable", "fixed", "safe.py", "repo_CWE"))
    assert "FILE review_target_001.py" in case["raw_prompt"]
    assert "FILE review_target_002.py" in case["raw_prompt"]
    assert case["sources"][0]["path"] == "CWE_001_vulnerable.py"
    assert case["sources"][0]["display_path"] == "review_target_001.py"


@pytest.mark.parametrize("damage", ["repository", "content", "path", "duplicate", "bytes"])
def test_invalid_corpus_fails_before_inference(tmp_path, damage):
    root = tmp_path / "src"
    root.mkdir()
    (root / "a.py").write_text("x = 1")
    (root / "b.py").write_text("x = 2")
    (tmp_path / "outside.py").write_text("x = 3")
    rows = [{"id": "a", "repository": "a", "split": "calibration", "files": ["a.py"]},
            {"id": "b", "repository": "b", "split": "test", "files": ["b.py"]}]
    if damage == "repository":
        rows[1]["repository"] = "a"
    elif damage == "content":
        rows[1]["files"] = ["a.py"]
    elif damage == "path":
        rows[0]["files"] = ["../outside.py"]
    elif damage == "duplicate":
        rows[1]["id"] = "a"
    with pytest.raises(ValueError):
        w.load_cases(manifest(tmp_path, rows), root, "calibration", 1 if damage == "bytes" else 100)


def test_reviews_replay_client_keeps_progress_and_token_provenance(tmp_path, checkpoint, monkeypatch):
    identity = checkpoint_identity(checkpoint)
    prepared = {"kind": "prepared_reviews", "checkpoint": identity, "split": "calibration",
                "prompt_format": w.PROMPT_FORMAT,
                "cases": [{"id": "a", "raw_prompt": "abc", "prompt_tokens": 3}]}
    source, baseline, replay = [tmp_path / name for name in ("input.json", "baseline.json", "replay.json")]
    write_new(source, prepared)
    mode = ["baseline"]
    monkeypatch.setattr(w, "check_server", lambda *args: ("model", {"mode": mode[0], "instance_id": "1"}))
    sent = []
    def complete(server, model, prompt, tokens, timeout):
        sent.append((prompt, tokens))
        return {"choices": [{"text": "review", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(prompt)}}, 1.0
    monkeypatch.setattr(w, "complete", complete)
    w.reviews(SimpleNamespace(input=source, output=baseline, server="local", max_tokens=10, timeout=1))
    assert read_json(baseline)["complete"]
    mode[0] = "statistics"
    monkeypatch.setattr(w, "tokenizer_tools", lambda path: (SimpleNamespace(encode=lambda text, **kwargs: list(text.encode())), None))
    w.replay(SimpleNamespace(model_dir=checkpoint, input=baseline, output=replay, server="local", timeout=1, max_seq_len=50))
    result = read_json(replay)
    assert result["complete"] and result["cases"][0]["token_sha256"] == w.token_digest(list(b"abcreview"))
    assert sent == [("abc", 10), ("abcreview", 1)]


def test_evaluation_uses_human_matches_and_counts_clean_false_positives(tmp_path):
    source, output = tmp_path / "graded.json", tmp_path / "metrics.json"
    case = {"id": "a", "label": "vulnerable", "expected_findings": [{"id": "one"}, {"id": "two"}],
            "judgment": {"baseline_matched_ids": ["one", "two"], "candidate_matched_ids": ["one"],
                         "baseline_false_positives": 0, "candidate_false_positives": 0}}
    clean = {"id": "b", "label": "clean", "expected_findings": [],
             "judgment": {"baseline_matched_ids": [], "candidate_matched_ids": [],
                          "baseline_false_positives": 0, "candidate_false_positives": 1}}
    write_new(source, {"kind": "paired_review", "checkpoint": {}, "cases": [case, clean]})
    w.evaluate(SimpleNamespace(input=source, output=output))
    metrics = read_json(output)["metrics"]
    assert metrics["baseline"]["recall"] == 1.0
    assert metrics["candidate"]["recall"] == 0.5
    assert metrics["candidate"]["precision"] == 0.5
    assert metrics["candidate"]["clean_case_false_positive_rate"] == 1.0


def test_ungraded_cases_do_not_produce_quality_claims(tmp_path):
    source = tmp_path / "ungraded.json"
    write_new(source, {"kind": "paired_review", "checkpoint": {}, "cases": [{
        "label": "clean", "expected_findings": [], "judgment": {
            "baseline_matched_ids": None, "baseline_false_positives": None}}]})
    with pytest.raises(ValueError, match="Fill every human judgment"):
        w.evaluate(SimpleNamespace(input=source, output=tmp_path / "metrics.json"))


def test_legacy_prepared_prompts_rejected_before_server_contact(tmp_path, monkeypatch):
    source = tmp_path / "legacy.json"
    write_new(source, {"kind": "prepared_reviews", "cases": [{"raw_prompt": "FILE vulnerable.c"}]})
    monkeypatch.setattr(w, "check_server", lambda *a: pytest.fail("must reject before contacting server"))
    with pytest.raises(ValueError, match="Re-run prepare"):
        w.reviews(SimpleNamespace(input=source))


def test_unrecognized_extensions_cannot_leak_labels():
    assert w.display_path("CWE_123/sample.vulnerable", 1) == "review_target_001.txt"
    assert w.display_path("fixed.CPP", 2) == "review_target_002.cpp"


def test_comparison_rejects_changed_prompts_and_sampling(tmp_path):
    case = {"id": "a", "raw_prompt": "source", "sources": [], "label": "clean", "expected_findings": [],
            "response": {"choices": [{"text": "review"}]}, "elapsed_seconds": 1}
    base = {"kind": "reviews", "complete": True, "checkpoint": {}, "split": "validation",
            "sampling": {"max_tokens": 10}, "server": {"mode": "baseline"}, "cases": [case]}
    candidate = {**base, "server": {"mode": "mask"}, "cases": [{**case, "raw_prompt": "different"}]}
    left, right = tmp_path / "base.json", tmp_path / "candidate.json"
    write_new(left, base)
    write_new(right, candidate)
    args = SimpleNamespace(baseline=left, candidate=right, output=tmp_path / "pairs.json")
    with pytest.raises(ValueError, match="source/prompt"):
        w.compare(args)
    candidate["cases"] = [case]
    candidate["sampling"] = {"max_tokens": 20}
    w.save_progress(right, candidate)
    with pytest.raises(ValueError, match="identical sampling"):
        w.compare(args)
    candidate["sampling"] = base["sampling"]
    w.save_progress(right, candidate)
    w.compare(args)
    assert read_json(args.output)["cases"][0]["sources"] == case["sources"]
