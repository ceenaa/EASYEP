from types import SimpleNamespace

import pytest
import torch

from freetoken.pruning.artifacts import read_json, verify_replay
from freetoken.pruning.runtime import (
    BatchCapture, PruningSession, active_capture, apply_expert_mask, validate_collection_config,
)
from freetoken.pruning.workflow import token_digest
from test_artifacts import checkpoint

G = {"n_layers": 2, "n_hash_layers": 1, "n_routed_experts": 4, "n_activated_experts": 2}


def config(root, output):
    return SimpleNamespace(model_path=str(root), expert_mask_path=None, expert_stats_path=str(output),
                           tp_info=SimpleNamespace(size=1), moe_strategy="offload", moe_cpu_layers=None,
                           max_running_req=1, cache_type="naive", cuda_graph_max_bs=0, cuda_graph_bs=None)


def observe(capture, layer, indices=None):
    ids = torch.tensor([[0, 1], [2, 1]]) if indices is None else indices
    gates = torch.tensor([[0.25, 0.75], [0.5, 0.5]])[:len(ids)]
    capture.begin_layer(layer, ids, gates)
    # Already weighted: route norms 5, 2 and 1, 3. Sensitivities are 1 and 0.
    capture.record_outputs(torch.tensor([[[3., 4.], [0., 2.]], [[1., 0.], [3., 0.]]])[:len(ids)])
    capture.set_routed(torch.ones(len(ids), 2))
    capture.finish_layer(layer, torch.tensor([[1., 0.], [1., 0.]])[:len(ids)],
                         torch.tensor([[0., 1.], [2., 0.]])[:len(ids)], torch.ones(len(ids), 1))


def test_weighted_norm_is_not_multiplied_by_gate_twice():
    capture = BatchCapture(G, 2)
    for layer in range(2):
        observe(capture, layer)
    score, gate, count = capture.result()
    torch.testing.assert_close(score, torch.tensor([[5., 2., 0., 0.]] * 2))
    torch.testing.assert_close(gate, torch.tensor([[0.25, 1.25, 0.5, 0.]] * 2))
    assert count.tolist() == [[1, 2, 1, 0]] * 2


def test_residual_expansion_matches_explicit_outer_product():
    capture = BatchCapture({**G, "n_layers": 1}, 3)
    ids = torch.tensor([[0, 1], [0, 2], [1, 3]])
    gates = torch.tensor([[0.2, 0.8], [0.5, 0.5], [0.7, 0.3]])
    down = torch.tensor([[[3., 4.], [0., 2.]], [[1., 0.], [0., 3.]], [[2., 1.], [1., 2.]]])
    post = torch.tensor([[3., 4., 0., 0.], [0., 0., 0., 2.], [0., 0., 0., 0.]])
    before = torch.tensor([[[1., 0.]] * 4] * 3)
    after = torch.tensor([[[0., 1.]] * 4] * 3)
    capture.begin_layer(0, ids, gates)
    capture.record_outputs(down)
    capture.finish_layer(0, before, after, post)
    expanded = post[:, None, :, None] * down[:, :, None, :]
    expected = torch.zeros(4)
    for token in range(3):
        for route in range(2):
            expected[ids[token, route]] += expanded[token, route].norm()
    torch.testing.assert_close(capture.result()[0][0], expected)
    torch.testing.assert_close(expected, torch.tensor([27., 10., 6., 0.]))


def test_mask_excludes_negative_biased_scores_and_preserves_all_ones():
    scores = torch.tensor([[-5., -4., -3., -2.]])
    allowed = torch.tensor([True, True, False, False])
    assert set(apply_expert_mask(scores, allowed).topk(2).indices[0].tolist()) == {0, 1}
    assert torch.equal(apply_expert_mask(scores, torch.ones(4, dtype=torch.bool)), scores)
    assert apply_expert_mask(scores, None) is scores


def test_logical_ids_are_snapshotted_before_slot_remap():
    capture = BatchCapture({**G, "n_layers": 1}, 1)
    ids = torch.tensor([[1, 3]])
    capture.begin_layer(0, ids, torch.ones(1, 2))
    ids.fill_(0)
    capture.record_outputs(torch.ones(1, 2, 2))
    capture.finish_layer(0, torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]]), torch.ones(1, 1))
    assert capture.result()[2].tolist() == [[0, 1, 0, 1]]


def test_chunked_aggregation_matches_whole_tensor():
    torch.manual_seed(0)
    n = 17
    ids = torch.randint(4, (n, 2))
    weights, down = torch.rand(n, 2), torch.randn(n, 2, 4)
    before, after = torch.randn(n, 4, 4), torch.randn(n, 4, 4)
    post = torch.rand(n, 4)
    def score(start, end):
        capture = BatchCapture({**G, "n_layers": 1}, end - start)
        capture.begin_layer(0, ids[start:end], weights[start:end])
        capture.record_outputs(down[start:end])
        capture.finish_layer(0, before[start:end], after[start:end], post[start:end])
        return capture.result()
    whole = score(0, n)
    chunks = list(zip(score(0, 5), score(5, n)))
    for expected, (a, b) in zip(whole, chunks):
        torch.testing.assert_close(expected, a + b)


@pytest.mark.parametrize("field,value", [("cache_type", "radix"), ("max_running_req", 2),
    ("moe_strategy", "hybrid"), ("moe_cpu_layers", "1"), ("cuda_graph_max_bs", 1),
    ("cuda_graph_bs", [1]), ("expert_mask_path", "mask.json")])
def test_unsupported_collection_is_rejected(checkpoint, tmp_path, field, value):
    args = config(checkpoint, tmp_path / "stats.json")
    setattr(args, field, value)
    with pytest.raises(ValueError):
        validate_collection_config(args)


def batch(uid=7, tokens=(10, 11), start=0, prefill=True):
    return SimpleNamespace(is_prefill=prefill, reqs=[SimpleNamespace(uid=uid)],
                           input_ids=torch.tensor(tokens), positions=torch.arange(start, start + len(tokens)))


def test_session_chunks_are_transactional_and_prove_replay_coverage(checkpoint, tmp_path):
    path = tmp_path / "stats.json"
    session = PruningSession(config(checkpoint, path))
    for start in (0, 2):
        with session.capture(batch(start=start)):
            assert active_capture() is not None
            for layer in range(2):
                observe(active_capture(), layer)
    assert active_capture() is None
    stats = read_json(path)
    assert stats["total_tokens"] == 4
    assert stats["requests"]["7"]["token_sha256"] == token_digest([10, 11, 10, 11])
    verify_replay(stats, {"checkpoint": session.identity, "complete": True,
                         "cases": [{"token_sha256": token_digest([10, 11, 10, 11]), "prompt_tokens": 4}]})
    with pytest.raises(RuntimeError, match="Incomplete"):
        with session.capture(batch(uid=8)):
            observe(active_capture(), 0)
    assert session.total_tokens == 4 and active_capture() is None
    assert read_json(path) == stats
    with pytest.raises(RuntimeError, match="Missing/repeated"):
        with session.capture(batch(start=0)):
            pass
    with session.capture(batch(prefill=False)):
        assert active_capture() is None
    with pytest.raises(RuntimeError, match="real request"):
        with session.capture(batch(uid=-1)):
            pass
    with pytest.raises(FileExistsError):
        PruningSession(config(checkpoint, path))


def test_missing_expert_output_observation_fails():
    capture = BatchCapture(G, 1)
    capture.begin_layer(0, torch.tensor([[0, 1]]), torch.ones(1, 2))
    with pytest.raises(RuntimeError, match="Missing per-expert"):
        capture.finish_layer(0, torch.ones(1, 2), torch.ones(1, 2), torch.ones(1, 1))


def test_live_capture_tracks_prefill_decode_and_flushes_final_token_coverage(checkpoint, tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("FREETOKEN_DISABLE_OVERLAP_SCHEDULING", "1")
    args = config(checkpoint, tmp_path / "live.json")
    args.expert_stats_decode = True
    session = PruningSession(args)
    with session.capture(batch()):
        for layer in range(2):
            observe(active_capture(), layer)
    for pos, token in ((2, 12), (3, 13)):
        with session.capture(batch(tokens=(token,), start=pos, prefill=False)):
            for layer in range(2):
                observe(active_capture(), layer, torch.tensor([[1, 3]]))
    session.finish_request(7, [10, 11, 12, 13, 14], "stop")
    stats = read_json(session.path)
    request = stats["requests"]["7"]
    assert request["tokens"] == 4
    assert request["prefill_tokens"] == request["decode_tokens"] == 2
    assert request["generated_token_ids"] == [12, 13, 14]
    assert request["finished"] and request["finish_reason"] == "stop"
    assert request["token_sha256"] == token_digest([10, 11, 12, 13])
    assert [sum(row) for row in stats["activation_counts"]] == [8, 8]
    events = [json.loads(line) for line in session.trace_path.read_text().splitlines()]
    assert [e["phase"] for e in events] == ["prefill", "decode", "decode"]
    assert [e["start_position"] for e in events] == [0, 2, 3]
    assert events[1]["expert_ids"] == [[[1, 3]], [[1, 3]]]
    assert events[1]["residual_weighted_output_norms"] == [[[5., 2.]], [[5., 2.]]]
    assert events[1]["sensitivity"] == [[1.], [1.]]
    with pytest.raises(ValueError, match="not calibration"):
        verify_replay(stats, {})
    with pytest.raises(RuntimeError, match="token IDs"):
        session.finish_request(7, [10, 11, 99, 13, 14], "stop")


def test_live_trace_requires_serial_eager_scheduler(checkpoint, tmp_path, monkeypatch):
    monkeypatch.delenv("FREETOKEN_DISABLE_OVERLAP_SCHEDULING", raising=False)
    args = config(checkpoint, tmp_path / "live.json")
    args.expert_stats_decode = True
    with pytest.raises(ValueError, match="OVERLAP"):
        validate_collection_config(args)
