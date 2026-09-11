#!/usr/bin/env python3
"""Read-only target inventory and mask validation; does not load model weights."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def run(argv):
    if not shutil.which(argv[0]):
        return {"available": False}
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return {"exit_code": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def inspect_model(folder):
    config_path = folder / "config.json"
    config = read_json(config_path) if config_path.exists() else {}
    model_type = config.get("model_type", "unknown")
    args_path = folder / "inference/config.json"
    if not args_path.exists():
        args_path = folder / "model_args.json"
    if model_type == "deepseek_v4" or args_path.exists():
        args = read_json(args_path)
        if "n_hash_layers" not in args:
            raise ValueError("Unrecognized inference config; cannot assume DeepSeek V4")
        model_type = "deepseek_v4"
        layer_ids = list(range(args["n_layers"]))
        experts, top_k = args["n_routed_experts"], args["n_activated_experts"]
        hash_ids = list(range(args["n_hash_layers"]))
        quantization = {key: args.get(key) for key in ("dtype", "expert_dtype", "scale_fmt")}
    elif model_type in ("deepseek_v2", "deepseek_v3"):
        first = config["first_k_dense_replace"]
        frequency = config.get("moe_layer_freq", 1)
        layer_ids = [i for i in range(first, config["num_hidden_layers"]) if i % frequency == 0]
        experts, top_k = config["n_routed_experts"], config["num_experts_per_tok"]
        hash_ids = []
        quantization = config.get("quantization_config", {})
    else:
        raise ValueError(f"Unsupported model_type {model_type!r}; need the exact MoE checkpoint config")
    if not layer_ids or not 0 < top_k <= experts:
        raise ValueError("Invalid MoE dimensions")
    index_path = folder / "model.safetensors.index.json"
    checkpoint = {}
    if index_path.exists():
        index = read_json(index_path)
        shards = sorted(set(index["weight_map"].values()))
        sizes = [((folder / name).stat().st_size if (folder / name).is_file() else None)
                 for name in shards]
        checkpoint = {
            "index_metadata_total_size_bytes": index.get("metadata", {}).get("total_size"),
            "shard_count": len(shards),
            "missing_shard_count": sum(size is None for size in sizes),
            "present_shards_bytes": sum(size for size in sizes if size is not None),
            "note": "Disk bytes and index metadata are not runtime RAM or VRAM requirements.",
        }
    return {"model_type": model_type, "architectures": config.get("architectures"),
            "moe_layer_ids": layer_ids, "experts_per_layer": experts,
            "active_experts_per_token": top_k, "hash_layer_ids": hash_ids,
            "quantization": quantization, "checkpoint": checkpoint}


def inspect_mask(path, model, model_dir=None):
    mask = read_json(path)
    if isinstance(mask, dict):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "FreeToken/python"))
        from freetoken.pruning.artifacts import checkpoint_identity, validate_mask

        rows = validate_mask(mask, checkpoint_identity(model_dir))
        return {"shape": [len(rows), len(rows[0])], "retained_per_layer": [sum(row) for row in rows],
                "checks": "version, configuration/index identity, shape, binary values, top-k capacity, hash preservation",
                "note": "Mask validated; CUDA behavior and quality still require measurement."}
    layer_ids = model["moe_layer_ids"]
    experts = model["experts_per_layer"]
    if not isinstance(mask, list) or len(mask) != len(layer_ids):
        raise ValueError(f"Mask needs {len(layer_ids)} rows in model MoE-layer order")
    counts = []
    for layer_id, row in zip(layer_ids, mask):
        if not isinstance(row, list) or len(row) != experts:
            raise ValueError(f"Layer {layer_id}: expected {experts} mask entries")
        if any(type(value) not in (int, float, bool) or value not in (0, 1) for value in row):
            raise ValueError(f"Layer {layer_id}: mask must contain only zero/one values")
        count = int(sum(row))
        if count < model["active_experts_per_token"]:
            raise ValueError(f"Layer {layer_id}: retained experts fewer than routing top-k")
        if layer_id in model["hash_layer_ids"] and count != experts:
            raise ValueError(f"Layer {layer_id}: hash-layer pruning needs a separate routing design")
        counts.append(count)
    return {"shape": [len(mask), experts], "retained_per_layer": counts,
            "checks": "shape, binary values, top-k capacity, unmodified V4 hash layers",
            "note": "Structural checks only. Does not prove checkpoint provenance, grouped routing capacity, quality, or runtime compatibility."}


def inspect_host(folder):
    versions = {}
    for package in ("freetoken", "torch", "triton", "transformers", "sglang"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    limits = {}
    for name in ("memory.max", "memory.current", "memory.high", "cpu.max",
                 "cpuset.cpus.effective", "memory/memory.limit_in_bytes",
                 "memory/memory.usage_in_bytes"):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            try:
                limits[name] = path.read_text().strip()
            except OSError:
                pass
    meminfo = {}
    path = Path("/proc/meminfo")
    if path.exists():
        for line in path.read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                meminfo[key] = value.strip()
    return {"platform": platform.system(), "machine": platform.machine(),
            "python": platform.python_version(), "packages": versions,
            "cpu_count": os.cpu_count(),
            "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "meminfo": meminfo, "cgroup_files_at_mount_root": limits,
            "allocation_note": "Confirm guaranteed RAM/CPU in the Vast offer. Host meminfo and root cgroup files may not reflect nested limits or the guaranteed baseline.",
            "disk_free_bytes": shutil.disk_usage(folder).free,
            "gpu": run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv"]),
            "cuda_toolkit": run(["nvcc", "--version"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--mask", type=Path)
    args = parser.parse_args()
    if args.mask and not args.model_dir:
        parser.error("--mask requires --model-dir")
    report = {"inference_verified": False}
    try:
        report["host"] = inspect_host(args.model_dir or Path.cwd())
        if args.model_dir:
            report["model"] = inspect_model(args.model_dir)
        if args.mask:
            report["mask"] = inspect_mask(args.mask, report["model"], args.model_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["error"] = str(exc)
    print(json.dumps(report, indent=2))
    return 2 if "error" in report else 0


if __name__ == "__main__":
    sys.exit(main())
