"""Small real-kernel checks to run on the rented GPU before loading the checkpoint."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")


@pytest.mark.parametrize("tokens", [1, 256])
def test_fp4_observer_sees_weighted_routes_without_changing_output(tokens):
    from freetoken.moe.fused_ds_fp4 import routed_experts_fp4_prefill
    from freetoken.pruning.runtime import _ACTIVE

    e, h, intermediate, k = 16, 256, 256, 6
    torch.manual_seed(42)
    x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16) * 0.1
    ids = torch.stack([torch.randperm(e, device="cuda")[:k] for _ in range(tokens)]).int().contiguous()
    weights = torch.rand(tokens, k, device="cuda")
    weights = (weights / weights.sum(-1, keepdim=True)).contiguous()
    banks = [torch.randint(0, 256, (e, 2 * intermediate, h // 2), dtype=torch.uint8, device="cuda"),
             torch.full((e, 2 * intermediate, h // 32), 121, dtype=torch.uint8, device="cuda"),
             torch.randint(0, 256, (e, h, intermediate // 2), dtype=torch.uint8, device="cuda"),
             torch.full((e, h, intermediate // 32), 121, dtype=torch.uint8, device="cuda")]
    def run():
        return routed_experts_fp4_prefill(x, ids, weights, *banks, 10.0, e)
    baseline = run()
    observed = []
    handle = _ACTIVE.set(SimpleNamespace(record_outputs=lambda down: observed.append(down.clone())))
    try:
        candidate = run()
    finally:
        _ACTIVE.reset(handle)
    assert len(observed) == 1
    assert observed[0].shape == (tokens, k, h)
    assert torch.equal(baseline, candidate)
    assert torch.equal(candidate, observed[0].sum(1))
    assert torch.isfinite(observed[0]).all()


def test_hyper_connection_reference_orientation():
    from freetoken.kernel.triton.dsv4.hc import hc_post_combine

    torch.manual_seed(4)
    routed = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(3, 4, 256, device="cuda", dtype=torch.bfloat16)
    post = torch.rand(3, 4, device="cuda")
    comb = torch.rand(3, 4, 4, device="cuda")
    result = hc_post_combine(routed, residual, post, comb)
    reference = torch.einsum("tpq,tpd->tqd", comb, residual.float()) + post[:, :, None] * routed[:, None, :].float()
    torch.testing.assert_close(result.float(), reference, atol=0.03125, rtol=0.008)


def test_actual_gate_all_ones_parity_and_exclusion():
    from freetoken.models.deepseek_v4.args import DeepseekV4Args
    from freetoken.models.deepseek_v4.moe import Gate

    args = DeepseekV4Args(dim=256, n_routed_experts=16, n_activated_experts=6, n_hash_layers=1)
    with torch.device("cuda"):
        gate = Gate(1, args)
    gate.weight.normal_(0, 0.01)
    gate.bias.fill_(-100)
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    tokens = torch.zeros(4, device="cuda", dtype=torch.long)
    weights, ids = gate.forward(x, tokens)
    gate._expert_mask = torch.ones(16, device="cuda", dtype=torch.bool)
    all_weights, all_ids = gate.forward(x, tokens)
    assert torch.equal(weights, all_weights) and torch.equal(ids, all_ids)
    gate._expert_mask[6:] = False
    weights, ids = gate.forward(x, tokens)
    assert (ids < 6).all()
    torch.testing.assert_close(weights.sum(-1), torch.full((4,), args.route_scale, device="cuda"))


def test_decode_block_finishes_live_observation_without_changing_output():
    from freetoken.models.deepseek_v4.model import Block
    from freetoken.pruning.runtime import BatchCapture, _ACTIVE, active_capture

    x = torch.randn(1, 1, 4, 4, device="cuda")
    post = torch.ones(1, 4, device="cuda")
    def ffn(value, input_ids):
        flat = value.reshape(1, 4)
        down = torch.stack((flat * 0.2, flat * 0.1), dim=1)
        routed = down.sum(1)
        capture = active_capture()
        if capture is not None:
            capture.begin_layer(0, torch.tensor([[0, 2]], device="cuda"),
                                torch.tensor([[0.4, 0.6]], device="cuda"))
            capture.record_outputs(down)
            capture.set_routed(routed)
        return (routed + flat * 0.1).view_as(value)
    block = SimpleNamespace(layer_id=0, hc_attn_fn=None, hc_attn_scale=None, hc_attn_base=None,
        hc_ffn_fn=None, hc_ffn_scale=None, hc_ffn_base=None,
        hc_pre=lambda value, *_: (value.mean(-2), post, None),
        hc_post=lambda value, residual, weights, _: residual + value.unsqueeze(-2) * weights.view(1, 1, 4, 1),
        attn_norm=SimpleNamespace(forward=lambda value: value),
        ffn_norm=SimpleNamespace(forward=lambda value: value),
        attn=SimpleNamespace(decode_step=lambda value, *_: value * 0.1),
        ffn=SimpleNamespace(forward=ffn))
    def run():
        return Block.decode_step(block, x, None, None, None, torch.tensor([10], device="cuda"))
    baseline = run()
    capture = BatchCapture({"n_layers": 1, "n_routed_experts": 4, "n_activated_experts": 2}, 1, trace=True)
    handle = _ACTIVE.set(capture)
    try:
        actual = run()
    finally:
        _ACTIVE.reset(handle)
    assert torch.equal(actual, baseline)
    assert capture.result()[2].tolist() == [[1, 0, 1, 0]]
    assert capture.trace_document()["expert_ids"] == [[[0, 2]]]
