#!/usr/bin/env python3
"""Immutable run provenance and content-bound stage checkpoints for EASY-EP V4.

The launcher deliberately keeps this logic outside its shell script so resume
validation can be unit tested.  A resume never replaces the provenance which
created the run: it computes a candidate record and requires exact equality.
Stage ``.done`` files are JSON attestations over the command, direct inputs, and
complete output trees; an empty or stale marker is therefore never skippable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
from typing import Any


MANIFEST_SCHEMA_VERSION = 3
STAGE_SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity(stat: os.stat_result) -> tuple[int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def digest_file(path: Path | str, *, include_path: bool = True) -> dict[str, Any]:
    """Hash one stable file observation and reject concurrent replacement."""
    path = Path(path)
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as fp:
        if _identity(os.fstat(fp.fileno())) != _identity(before):
            raise RuntimeError(f"file changed while opening for hashing: {path}")
        for block in iter(lambda: fp.read(16 * 1024 * 1024), b""):
            h.update(block)
        finished = os.fstat(fp.fileno())
    after = path.stat()
    if _identity(before) != _identity(finished) or _identity(before) != _identity(after):
        raise RuntimeError(f"file changed while hashing: {path}")
    record: dict[str, Any] = {"size": finished.st_size, "sha256": h.hexdigest()}
    if include_path:
        record["path"] = str(path.resolve())
    return record


def digest_tree(root: Path | str) -> dict[str, Any]:
    """Commit to every relative path and byte in a symlink-free directory tree."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"required directory is missing: {root}")
    aggregate = hashlib.sha256()
    count = total = 0
    inventory = sorted(root.rglob("*"))
    for path in inventory:
        if path.is_symlink():
            raise ValueError(f"provenance trees may not contain symlinks: {path}")
        if not path.is_file():
            continue
        item = digest_file(path)
        relative = path.relative_to(root).as_posix()
        aggregate.update(relative.encode("utf-8") + b"\0")
        aggregate.update(item["sha256"].encode("ascii") + b"\n")
        count += 1
        total += item["size"]
    after_inventory = sorted(root.rglob("*"))
    if [path.relative_to(root) for path in inventory] != [
            path.relative_to(root) for path in after_inventory]:
        raise RuntimeError(f"directory tree changed while hashing: {root}")
    return {
        "root": str(root),
        "file_count": count,
        "total_bytes": total,
        "tree_sha256": aggregate.hexdigest(),
    }


def digest_path(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"stage paths may not be symlinks: {path}")
    if path.is_file():
        return {"kind": "file", **digest_file(path)}
    if path.is_dir():
        return {"kind": "tree", **digest_tree(path)}
    raise ValueError(f"required stage path is missing: {path}")


def checkpoint_layout(checkpoint: Path, nproc: int) -> dict[str, Any]:
    records = []
    for path in sorted(checkpoint.glob(f"model*-mp{nproc}.safetensors")):
        stat = path.stat()
        records.append({
            "path": str(path.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {"files": records, "layout_sha256": hashlib.sha256(encoded).hexdigest()}


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("torch", "transformers", "safetensors", "tilelang",
                 "fast-hadamard-transform"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def environment_record(venv: Path | str) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "python_module": os.environ.get("PYTHON_MODULE"),
        "cuda_module": os.environ.get("CUDA_MODULE"),
        "loaded_modules": os.environ.get("LOADEDMODULES", "").split(":"),
        "venv": str(Path(venv).resolve()),
        "determinism": {
            name: os.environ.get(name) for name in (
                "PYTHONHASHSEED", "OMP_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING", "STRICT_PINS",
            )
        },
    }


def _runtime_file(role: str, path: Path | str, *, portable_path: bool = False) -> dict[str, Any]:
    return {"role": role, **digest_file(path, include_path=not portable_path)}


def build_provenance(a: argparse.Namespace) -> dict[str, Any]:
    repo = Path(a.repo).resolve()
    code = Path(a.code).resolve()
    checkpoint = Path(a.checkpoint).resolve()
    data_root = Path(a.data_root).resolve()
    sidecar_record = _runtime_file("checkpoint_provenance", a.checkpoint_provenance)
    sidecar = json.loads(Path(a.checkpoint_provenance).read_text(encoding="utf-8"))
    if _runtime_file("checkpoint_provenance", a.checkpoint_provenance) != sidecar_record:
        raise RuntimeError("checkpoint provenance sidecar changed while reading")
    content_identity = sidecar.get("content_identity_sha256")
    if not isinstance(content_identity, str) or SHA256_RE.fullmatch(content_identity) is None:
        raise ValueError("verified checkpoint sidecar lacks a portable content identity")

    runtime_files = [
        _runtime_file("instrumentation", a.script),
        # Slurm changes the spool path on every submission.  Its bytes, not that
        # volatile path, identify the entrypoint which is actually executing.
        _runtime_file("executing_entrypoint", a.entrypoint, portable_path=True),
        _runtime_file("config", a.config),
        _runtime_file("compatibility_entrypoint", repo / "v4/easyep.sbatch"),
        _runtime_file("requirements", repo / "v4/requirements-v4.txt"),
        _runtime_file("checkpoint_helper", repo / "v4/checkpoint_provenance.py"),
        _runtime_file("run_provenance_helper", repo / "v4/run_provenance.py"),
        sidecar_record,
        _runtime_file("tokenizer", checkpoint / "tokenizer.json"),
        _runtime_file("tokenizer_config", checkpoint / "tokenizer_config.json"),
    ]
    return {
        "source": {
            "git_sha": a.git_sha,
            "git_tracked_dirty": a.git_tracked_dirty,
            "runtime_files": runtime_files,
            "official_inference_tree": digest_tree(code),
            "official_encoding_tree": digest_tree(code.parent / "encoding"),
        },
        "model": {
            "declared_id": a.model_id,
            "declared_revision": a.model_revision,
            "declared_inference_revision": a.inference_revision,
            "checkpoint_root": str(checkpoint),
            "checkpoint_content_identity_sha256": content_identity,
            "checkpoint_provenance": sidecar_record,
            "checkpoint_layout": checkpoint_layout(checkpoint, a.nproc_per_node),
        },
        "inputs": {
            "questions": digest_file(data_root / "questions_used.json"),
            "pair_manifest": digest_file(
                data_root / "vulnerable-js-files/CODEQL_SECURE_MANIFEST.jsonl"),
            "calibration_and_pair_tree": digest_tree(data_root / "vulnerable-js-files"),
        },
        "environment": environment_record(a.venv),
        "parameters": {
            "keep": a.keep, "n_calib": a.n_calib, "n_pairs": a.n_pairs,
            "max_seq_len": a.max_seq_len, "max_new_tokens": a.max_new_tokens,
            "pair_max_new_tokens": a.pair_max_new_tokens,
            "validate_n_calib": a.n_calib, "max_chunks": a.max_chunks,
            "question_limit": a.question_limit, "seed": a.seed,
            "temperature": a.temperature, "nproc_per_node": a.nproc_per_node,
        },
    }


def first_difference(old: Any, new: Any, path: str = "provenance") -> str:
    if type(old) is not type(new):
        return f"{path}: type {type(old).__name__} != {type(new).__name__}"
    if isinstance(old, dict):
        if set(old) != set(new):
            missing = sorted(set(old) - set(new))
            added = sorted(set(new) - set(old))
            return f"{path}: keys changed (missing={missing}, added={added})"
        for key in sorted(old):
            difference = first_difference(old[key], new[key], f"{path}.{key}")
            if difference:
                return difference
        return ""
    if isinstance(old, list):
        if len(old) != len(new):
            return f"{path}: length {len(old)} != {len(new)}"
        for index, value in enumerate(old):
            difference = first_difference(value, new[index], f"{path}[{index}]")
            if difference:
                return difference
        return ""
    return "" if old == new else f"{path}: {old!r} != {new!r}"


def read_manifest(path: Path | str) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("run manifest has an unsupported schema; start a new RUN_ID")
    provenance = manifest.get("provenance")
    digest = manifest.get("provenance_sha256")
    if not isinstance(provenance, dict) or digest != canonical_sha256(provenance):
        raise ValueError("run manifest provenance is missing or internally inconsistent")
    return manifest


def verify_resume(existing: dict[str, Any], candidate: dict[str, Any]) -> None:
    if existing.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("run manifest has an unsupported schema; start a new RUN_ID")
    old = existing.get("provenance")
    old_digest = existing.get("provenance_sha256")
    if not isinstance(old, dict) or old_digest != canonical_sha256(old):
        raise ValueError("existing run manifest provenance is invalid")
    difference = first_difference(old, candidate)
    if difference:
        raise ValueError(f"resume provenance mismatch: {difference}")


def atomic_write_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(value, fp, indent=2, sort_keys=True)
            fp.write("\n")
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def attempt_record(kind: str, started_utc: str, return_code: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": kind, "started_utc": started_utc,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", "manual"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "node": platform.node(),
        },
    }
    if return_code is not None:
        record.update({
            "finished_utc": utc_now(), "exit_code": return_code,
            "status": "complete" if return_code == 0 else "failed",
        })
    return record


def initialize_or_verify_manifest(path: Path | str, candidate: dict[str, Any], *,
                                  resume: bool, attempt_started_utc: str) -> None:
    """Publish a new manifest, or verify a resume without rewriting one byte."""
    path = Path(path)
    if resume:
        existing = json.loads(path.read_text(encoding="utf-8"))
        verify_resume(existing, candidate)
        print(f"resume provenance exactly matches {path}")
        return
    if path.exists():
        raise ValueError(f"refusing to replace existing run manifest: {path}")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": attempt_started_utc,
        "provenance": candidate,
        "provenance_sha256": canonical_sha256(candidate),
    }
    atomic_write_json(path, manifest)
    print(f"immutable run provenance initialized: {path}")


def cmd_init(a: argparse.Namespace) -> None:
    initialize_or_verify_manifest(
        a.manifest, build_provenance(a), resume=a.resume,
        attempt_started_utc=a.attempt_started_utc,
    )


def _find_runtime(provenance: dict[str, Any], role: str) -> dict[str, Any]:
    matches = [item for item in provenance["source"]["runtime_files"]
               if item.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"manifest has {len(matches)} runtime records for {role}")
    return matches[0]


def cmd_preflight(a: argparse.Namespace) -> None:
    manifest = read_manifest(a.manifest)
    provenance = manifest["provenance"]
    for expected in provenance["source"]["runtime_files"]:
        role = expected["role"]
        if role == "executing_entrypoint":
            current = _runtime_file(role, a.entrypoint, portable_path=True)
        else:
            current = _runtime_file(role, expected["path"])
        if current != expected:
            raise RuntimeError(f"runtime file changed after provenance creation: {role}")
    for key in ("official_inference_tree", "official_encoding_tree"):
        expected = provenance["source"][key]
        if digest_tree(expected["root"]) != expected:
            raise RuntimeError(f"official source tree changed after provenance creation: {key}")
    for key in ("questions", "pair_manifest"):
        expected = provenance["inputs"][key]
        if digest_file(expected["path"]) != expected:
            raise RuntimeError(f"experiment input changed after provenance creation: {key}")
    expected_tree = provenance["inputs"]["calibration_and_pair_tree"]
    if digest_tree(expected_tree["root"]) != expected_tree:
        raise RuntimeError("calibration/pair input tree changed after provenance creation")
    expected_sidecar = provenance["model"]["checkpoint_provenance"]
    observed_sidecar = {
        "role": "checkpoint_provenance", **digest_file(expected_sidecar["path"])
    }
    if observed_sidecar != expected_sidecar:
        raise RuntimeError("checkpoint provenance sidecar changed after provenance creation")
    sidecar = json.loads(Path(expected_sidecar["path"]).read_text(encoding="utf-8"))
    if {"role": "checkpoint_provenance",
            **digest_file(expected_sidecar["path"])} != expected_sidecar:
        raise RuntimeError("checkpoint provenance sidecar changed while reading")
    if sidecar.get("content_identity_sha256") != provenance["model"][
            "checkpoint_content_identity_sha256"]:
        raise RuntimeError("checkpoint content identity changed after provenance creation")
    current_layout = checkpoint_layout(
        Path(provenance["model"]["checkpoint_root"]),
        provenance["parameters"]["nproc_per_node"],
    )
    if current_layout != provenance["model"]["checkpoint_layout"]:
        raise RuntimeError("checkpoint layout changed after provenance creation")
    if environment_record(provenance["environment"]["venv"]) != provenance["environment"]:
        raise RuntimeError("runtime environment changed after provenance creation")
    print("run inputs still match immutable provenance")


def stage_observation(a: argparse.Namespace) -> dict[str, Any]:
    manifest = read_manifest(a.manifest)
    return {
        "stage_schema_version": STAGE_SCHEMA_VERSION,
        "stage": a.stage,
        "run_provenance_sha256": manifest["provenance_sha256"],
        "command": list(a.command_arg or []),
        "command_sha256": canonical_sha256(list(a.command_arg or [])),
        "inputs": [digest_path(path) for path in (a.input or [])],
        "outputs": [digest_path(path) for path in (a.output or [])],
    }


def cmd_mark_stage(a: argparse.Namespace) -> None:
    record = stage_observation(a)
    record["completed_utc"] = utc_now()
    atomic_write_json(a.marker, record)
    print(f"content-bound stage marker written: {a.marker}")


def cmd_verify_stage(a: argparse.Namespace) -> None:
    marker = json.loads(Path(a.marker).read_text(encoding="utf-8"))
    if not isinstance(marker, dict):
        raise ValueError(f"stage marker is not an object: {a.marker}")
    observed = {key: value for key, value in marker.items() if key != "completed_utc"}
    expected = stage_observation(a)
    difference = first_difference(observed, expected, "stage_marker")
    if difference:
        raise ValueError(f"stage marker mismatch for {a.stage}: {difference}")
    print(f"stage inputs and outputs verified: {a.stage}")


def cmd_finalize(a: argparse.Namespace) -> None:
    manifest = read_manifest(a.manifest)
    return_code = a.return_code
    status_path = Path(a.status)
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if (not isinstance(status, dict)
                or status.get("status_schema_version") != 1
                or status.get("run_provenance_sha256") != manifest["provenance_sha256"]
                or not isinstance(status.get("attempts", []), list)):
            raise ValueError("run status does not belong to this immutable manifest")
    else:
        status = {
            "status_schema_version": 1,
            "run_provenance_sha256": manifest["provenance_sha256"],
            "attempts": [],
        }
    attempts = list(status["attempts"])
    attempts.append(attempt_record(
        "resume" if a.resumed else "initial", a.attempt_started_utc, return_code))
    status.update({
        "status": "complete" if return_code == 0 else "failed",
        "exit_code": return_code,
        "finished_utc": utc_now(),
        "attempts": attempts,
        "stages_completed": list(a.stage or []),
        "artifacts": sorted(
            str(item.relative_to(a.out_dir)) for item in Path(a.out_dir).rglob("*")
            if item.is_file() and item not in (Path(a.manifest), status_path)
        ),
    })
    atomic_write_json(status_path, status)


def add_stage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--command-arg", action="append", default=[])
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--output", action="append", default=[])


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    for name in ("manifest", "repo", "script", "entrypoint", "config", "code",
                 "checkpoint", "checkpoint_provenance", "data_root", "venv",
                 "git_sha", "git_tracked_dirty", "model_id", "model_revision",
                 "inference_revision", "attempt_started_utc"):
        init.add_argument(f"--{name.replace('_', '-')}", required=True)
    for name in ("keep", "n_calib", "n_pairs", "max_seq_len", "max_new_tokens",
                 "pair_max_new_tokens", "max_chunks", "question_limit", "seed",
                 "nproc_per_node"):
        init.add_argument(f"--{name.replace('_', '-')}", type=int, required=True)
    init.add_argument("--temperature", type=float)
    init.add_argument("--resume", action="store_true")
    init.set_defaults(func=cmd_init)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--manifest", required=True)
    preflight.add_argument("--entrypoint", required=True)
    preflight.set_defaults(func=cmd_preflight)

    mark = sub.add_parser("mark-stage")
    add_stage_arguments(mark)
    mark.set_defaults(func=cmd_mark_stage)
    verify = sub.add_parser("verify-stage")
    add_stage_arguments(verify)
    verify.set_defaults(func=cmd_verify_stage)

    final = sub.add_parser("finalize")
    final.add_argument("--manifest", required=True)
    final.add_argument("--status", required=True)
    final.add_argument("--out-dir", required=True)
    final.add_argument("--return-code", type=int, required=True)
    final.add_argument("--attempt-started-utc", required=True)
    final.add_argument("--resumed", action="store_true")
    final.add_argument("--stage", action="append", default=[])
    final.set_defaults(func=cmd_finalize)
    return root


def main() -> None:
    a = parser().parse_args()
    try:
        a.func(a)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"run provenance error: {exc}") from exc


if __name__ == "__main__":
    main()
