#!/usr/bin/env python3
"""Launch the local FreeToken checkout with conservative first-experiment settings."""

import argparse
import os
from pathlib import Path
import platform
import shlex
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "FreeToken/python"))

from freetoken.pruning.artifacts import checkpoint_identity, read_json, validate_mask


def build_command(args):
    model = Path(args.model_dir).resolve(strict=True)
    identity = checkpoint_identity(model)
    if args.phase == "baseline" and args.artifact:
        raise ValueError("Baseline takes no artifact")
    if args.phase != "baseline" and not args.artifact:
        raise ValueError("Statistics/mask mode requires --artifact")
    if getattr(args, "trace_decode", False) and args.phase != "statistics":
        raise ValueError("Live decode tracing requires statistics mode")
    if args.max_seq_len <= 0 or args.max_seq_len % 128 or not 0 < args.chunk_tokens <= args.max_seq_len:
        raise ValueError("Sequence length must be a positive multiple of 128; chunk must fit it")
    if args.cache_slots < 2 * identity["geometry"]["n_routed_experts"]:
        raise ValueError("This launch profile reserves at least two expert layers for prefill")
    if not 0 < args.port < 65536:
        raise ValueError("Invalid port")
    command = ["ft", "serve", "--model", str(model), "--host", "127.0.0.1", "--port", str(args.port),
               "--moe-strategy", "offload", "--expert-load", "serial",
               "--moe-cache-size", str(args.cache_slots), "--max-running-requests", "1",
               "--cache-type", "naive", "--cuda-graph-max-bs", "0",
               "--max-prefill-length", str(args.chunk_tokens), "--max-seq-len-override", str(args.max_seq_len),
               "--num-tokens", str(args.max_seq_len), "--memory-ratio", "0.85"]
    # Slurm can expose a MIG UUID that the physical-GPU selector cannot resolve.
    if not getattr(args, "inherit_gpu", False):
        command += ["--gpu", "0"]
    if args.artifact:
        artifact = Path(args.artifact).resolve()
        if args.phase == "mask":
            validate_mask(read_json(artifact), identity)
            command += ["--expert-mask", str(artifact)]
        else:
            if artifact.exists():
                raise FileExistsError("Use a new statistics output for each server run")
            command += ["--expert-stats", str(artifact)]
            if getattr(args, "trace_decode", False):
                command += ["--expert-stats-decode"]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["baseline", "statistics", "mask"])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--artifact")
    parser.add_argument("--trace-decode", action="store_true")
    parser.add_argument("--inherit-gpu", action="store_true",
                        help="Use CUDA's first visible device without physical-GPU selection (Slurm/MIG)")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--cache-slots", type=int, default=512)
    parser.add_argument("--port", type=int, default=1919)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command = build_command(args)
    print("FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1 " + shlex.join(command), flush=True)
    if args.dry_run:
        return
    if platform.system() != "Linux" or not shutil.which("nvidia-smi") or not shutil.which("nvcc"):
        raise RuntimeError("Run on Linux with the NVIDIA driver and CUDA toolkit installed")
    executable = shutil.which(command[0])
    if executable is None:
        raise RuntimeError("Activate the environment containing the editable FreeToken install")
    model = Path(args.model_dir).resolve()
    index = read_json(model / "model.safetensors.index.json")
    missing = [name for name in set(index["weight_map"].values()) if not (model / name).is_file()]
    if missing:
        raise RuntimeError(f"Checkpoint is incomplete: {len(missing)} missing shards")
    env = dict(os.environ)
    env["FREETOKEN_DISABLE_OVERLAP_SCHEDULING"] = "1"
    env["PYTHONPATH"] = str(ROOT / "FreeToken/python") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    os.execvpe(executable, command, env)


if __name__ == "__main__":
    main()
