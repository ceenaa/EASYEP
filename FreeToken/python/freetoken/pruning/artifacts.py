"""Checkpoint-bound pruning artifacts. This module needs no GPU or torch."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from . import ADAPTER_VERSION, METRIC


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def checkpoint_identity(model_dir):
    root = Path(model_dir)
    names = ("config.json", "inference/config.json", "model.safetensors.index.json")
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    config = read_json(root / "config.json")
    args = read_json(root / "inference/config.json")
    if config.get("model_type") != "deepseek_v4":
        raise ValueError("Expert pruning currently supports DeepSeek-V4 only")
    geometry = {key: args[key] for key in (
        "n_layers", "n_hash_layers", "n_routed_experts", "n_activated_experts", "dim", "moe_inter_dim"
    )}
    if any(type(v) is not int or v < 0 for v in geometry.values()):
        raise ValueError("Invalid model geometry")
    if not (0 < geometry["n_activated_experts"] <= geometry["n_routed_experts"]
            and 0 <= geometry["n_hash_layers"] < geometry["n_layers"]):
        raise ValueError("Invalid expert counts")
    return {"model_type": "deepseek_v4", "config_index_sha256": hashes, "geometry": geometry}


def validate_mask(document, identity):
    if document.get("adapter_version") != ADAPTER_VERSION or document.get("kind") != "expert_mask":
        raise ValueError("Expected a versioned FreeToken expert-mask artifact")
    if document.get("checkpoint") != identity:
        raise ValueError("Mask checkpoint configuration/index identity does not match")
    if document.get("metric") not in (None, METRIC):
        raise ValueError("Unsupported mask metric; collect fresh statistics with the current scorer")
    geometry = identity["geometry"]
    layers, experts = geometry["n_layers"], geometry["n_routed_experts"]
    rows = document.get("mask")
    if not isinstance(rows, list) or len(rows) != layers:
        raise ValueError(f"Expected {layers} mask rows")
    for layer, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != experts:
            raise ValueError(f"Layer {layer}: expected {experts} mask entries")
        if any(type(value) not in (bool, int, float) or value not in (0, 1) for value in row):
            raise ValueError(f"Layer {layer}: mask must be binary")
        if sum(row) < geometry["n_activated_experts"]:
            raise ValueError(f"Layer {layer}: fewer retained experts than routing top-k")
        if layer < geometry["n_hash_layers"] and not all(row):
            raise ValueError(f"Layer {layer}: hash-routed experts must remain intact")
    return rows


def verify_replay(stats, replay):
    if stats.get("collection") == "live_prefill_decode":
        raise ValueError("Live smoke traces are not calibration replay artifacts")
    if (stats.get("kind") != "expert_statistics" or stats.get("metric") != METRIC
            or stats.get("adapter_version") != ADAPTER_VERSION):
        raise ValueError("Unsupported scoring artifact")
    if replay.get("checkpoint") != stats.get("checkpoint") or replay.get("complete") is not True:
        raise ValueError("Need a completed replay log for this checkpoint")
    expected = sorted((case["token_sha256"], case["prompt_tokens"]) for case in replay["cases"])
    actual = sorted((case["token_sha256"], case["tokens"]) for case in stats["requests"].values())
    if not expected or actual != expected:
        raise ValueError("Collected token coverage does not match the complete replay log")
    if stats["total_tokens"] != sum(count for _, count in expected):
        raise ValueError("Inconsistent total token count")
    geometry = stats["checkpoint"]["geometry"]
    counts = stats.get("activation_counts", [])
    if len(counts) != geometry["n_layers"]:
        raise ValueError("Missing per-layer activation coverage")
    for row in counts:
        if (len(row) != geometry["n_routed_experts"]
                or any(type(n) is not int or n < 0 for n in row)
                or sum(row) != geometry["n_activated_experts"] * stats["total_tokens"]):
            raise ValueError("Per-layer activation counts disagree with replay coverage")


def make_mask(stats, replay, keep):
    verify_replay(stats, replay)
    identity = stats["checkpoint"]
    g = identity["geometry"]
    if type(keep) is not int or not g["n_activated_experts"] <= keep <= g["n_routed_experts"]:
        raise ValueError("Retained expert count must lie between top-k and the full pool")
    rows, rankings = [], []
    scores = stats["scores"]
    if len(scores) != g["n_layers"]:
        raise ValueError("Incomplete layer scores")
    for layer, values in enumerate(scores):
        if len(values) != g["n_routed_experts"] or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"Layer {layer}: invalid importance scores")
        if layer >= g["n_hash_layers"] and sum(values) <= 0:
            raise ValueError(f"Layer {layer}: no positive importance evidence")
        order = sorted(range(len(values)), key=lambda expert: (-values[expert], expert))
        retained = set(order[:keep]) if layer >= g["n_hash_layers"] else set(order)
        rows.append([int(expert in retained) for expert in range(len(values))])
        rankings.append(order)
    result = {"adapter_version": ADAPTER_VERSION, "kind": "expert_mask", "metric": METRIC,
              "checkpoint": identity, "keep_learned": keep, "mask": rows,
              "ranking": rankings, "calibration_tokens": stats["total_tokens"],
              "collector_provenance": stats.get("provenance"),
              "replay_sha256": hashlib.sha256(json.dumps(replay, sort_keys=True).encode()).hexdigest()}
    validate_mask(result, identity)
    return result


def all_ones_mask(identity):
    g = identity["geometry"]
    return {"adapter_version": ADAPTER_VERSION, "kind": "expert_mask", "checkpoint": identity,
            "mask": [[1] * g["n_routed_experts"] for _ in range(g["n_layers"])]}


def server_metadata(config):
    try:
        identity = checkpoint_identity(config.model_path)
    except (OSError, ValueError, KeyError):
        return {"supported": False}
    mask = getattr(config, "expert_mask_path", None)
    stats = getattr(config, "expert_stats_path", None)
    metadata = {"supported": True, "checkpoint": identity,
            "mode": "statistics" if stats else "mask" if mask else "baseline",
            "mask_sha256": hashlib.sha256(Path(mask).read_bytes()).hexdigest() if mask else None}
    if stats:
        metadata["collection"] = "live_prefill_decode" if getattr(config, "expert_stats_decode", False) else "prefill_only_replay"
    return metadata


def write_new(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(document, indent=2, allow_nan=False) + "\n"
    with path.open("x") as handle:
        handle.write(serialized)
