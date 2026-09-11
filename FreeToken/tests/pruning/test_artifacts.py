import copy
import json

import pytest

from freetoken.pruning.artifacts import (
    all_ones_mask, checkpoint_identity, make_mask, validate_mask, verify_replay,
)
from freetoken.pruning import ADAPTER_VERSION, METRIC


@pytest.fixture
def checkpoint(tmp_path):
    root = tmp_path / "model"
    (root / "inference").mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"model_type": "deepseek_v4"}))
    (root / "inference/config.json").write_text(json.dumps({
        "n_layers": 2, "n_hash_layers": 1, "n_routed_experts": 4,
        "n_activated_experts": 2, "dim": 4, "moe_inter_dim": 8,
    }))
    (root / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    return root


def example(identity):
    stats = {"adapter_version": ADAPTER_VERSION, "metric": METRIC,
             "kind": "expert_statistics", "checkpoint": identity,
             "scores": [[0, 0, 0, 0], [1, 4, 4, 0]], "total_tokens": 2,
             "activation_counts": [[1, 1, 1, 1], [1, 1, 1, 1]],
             "requests": {"7": {"token_sha256": "abc", "tokens": 2}}}
    replay = {"checkpoint": identity, "complete": True,
              "cases": [{"token_sha256": "abc", "prompt_tokens": 2}]}
    return stats, replay


def test_mask_ranking_preserves_hash_and_breaks_ties(checkpoint):
    identity = checkpoint_identity(checkpoint)
    stats, replay = example(identity)
    mask = make_mask(stats, replay, 2)
    assert mask["mask"] == [[1, 1, 1, 1], [0, 1, 1, 0]]
    assert mask["ranking"][1] == [1, 2, 0, 3]
    validate_mask(mask, identity)


@pytest.mark.parametrize("damage", ["hash", "capacity", "shape", "binary", "identity", "version"])
def test_invalid_masks_fail_closed(checkpoint, damage):
    identity = checkpoint_identity(checkpoint)
    mask = all_ones_mask(identity)
    if damage == "hash":
        mask["mask"][0][0] = 0
    elif damage == "capacity":
        mask["mask"][1] = [1, 0, 0, 0]
    elif damage == "shape":
        mask["mask"].pop()
    elif damage == "binary":
        mask["mask"][1][0] = 0.1
    elif damage == "identity":
        mask["checkpoint"] = {}
    else:
        mask["adapter_version"] = -1
    with pytest.raises(ValueError):
        validate_mask(mask, identity)


def test_identity_changes_with_index(checkpoint):
    before = checkpoint_identity(checkpoint)
    (checkpoint / "model.safetensors.index.json").write_text('{"weight_map": {}, "metadata": {}}')
    with pytest.raises(ValueError):
        validate_mask(all_ones_mask(before), checkpoint_identity(checkpoint))


@pytest.mark.parametrize("damage", ["incomplete", "extra_request", "missing_request", "hash", "total", "metric", "counts"])
def test_replay_rejects_incomplete_or_contaminated_stats(checkpoint, damage):
    stats, replay = example(checkpoint_identity(checkpoint))
    if damage == "incomplete":
        replay["complete"] = False
    elif damage == "extra_request":
        stats["requests"]["8"] = copy.deepcopy(stats["requests"]["7"])
    elif damage == "missing_request":
        stats["requests"] = {}
    elif damage == "hash":
        stats["requests"]["7"]["token_sha256"] = "other"
    elif damage == "total":
        stats["total_tokens"] += 1
    elif damage == "counts":
        stats["activation_counts"][1][0] = 0
    else:
        stats["metric"] = "activation_count"
    with pytest.raises(ValueError):
        verify_replay(stats, replay)


@pytest.mark.parametrize("values", [[0, 0, 0, 0], [1, -1, 0, 0], [float("nan"), 1, 2, 3]])
def test_unusable_scores_cannot_generate_masks(checkpoint, values):
    stats, replay = example(checkpoint_identity(checkpoint))
    stats["scores"][1] = values
    with pytest.raises(ValueError):
        make_mask(stats, replay, 2)


def test_old_metric_requires_fresh_statistics_and_mask(checkpoint):
    identity = checkpoint_identity(checkpoint)
    stats, replay = example(identity)
    stats["metric"] = "easyep_v4_hc_weighted_v1"
    with pytest.raises(ValueError, match="Unsupported scoring"):
        make_mask(stats, replay, 2)
    mask = all_ones_mask(identity)
    mask["metric"] = stats["metric"]
    with pytest.raises(ValueError, match="Unsupported mask metric"):
        validate_mask(mask, identity)
