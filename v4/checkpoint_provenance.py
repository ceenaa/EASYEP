#!/usr/bin/env python3
"""Create and verify a content-addressed provenance record for V4 checkpoints.

Converted V4 checkpoints are hundreds of GiB, so hashing every shard at the
start of every Slurm job is wasteful.  ``create`` performs that full read once;
``verify`` checks the content-addressed record plus exact shard metadata cheaply.
Pass ``--full`` to ``verify`` for a fresh byte-for-byte audit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, NoReturn


SCHEMA_VERSION = 1
DEFAULT_NAME = "EASYEP_CHECKPOINT_PROVENANCE.json"
PINNED_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
PINNED_REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
PINNED_CONFIG_SHA256 = "c90861f3d10a9e4ef5954f8f1a34c529d480da1c5799f84660028f4e38e14e71"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_SOURCE_FILES = (
    "inference/convert.py",
    "inference/config.json",
    "inference/model.py",
    "inference/kernel.py",
    "inference/generate.py",
    "encoding/encoding_dsv4.py",
)


def fail(message: str) -> NoReturn:
    raise SystemExit(f"checkpoint provenance error: {message}")


def file_hash_observation(path: Path, block_bytes: int = 16 * 1024 * 1024,
                          include_git_blob: bool = False
                          ) -> tuple[dict[str, str], os.stat_result]:
    """Hash one opened inode and return the metadata from that same observation."""
    if not path.is_file():
        fail(f"file not found: {path}")
    before = path.stat()
    identity = lambda stat: (
        stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
    )
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1() if include_git_blob else None
    if git_sha1 is not None:
        # Git/Hugging Face ETag compatibility, not a security primitive.
        git_sha1.update(f"blob {before.st_size}\0".encode("ascii"))
    with path.open("rb") as fp:
        if identity(os.fstat(fp.fileno())) != identity(before):
            fail(f"file changed while opening for hashing: {path}")
        for block in iter(lambda: fp.read(block_bytes), b""):
            sha256.update(block)
            if git_sha1 is not None:
                git_sha1.update(block)
        finished = os.fstat(fp.fileno())
    after = path.stat()
    if identity(before) != identity(finished) or identity(before) != identity(after):
        fail(f"file changed while hashing: {path}")
    hashes = {"sha256": sha256.hexdigest()}
    if git_sha1 is not None:
        hashes["git_blob_sha1"] = git_sha1.hexdigest()
    return hashes, finished


def sha256_file_observation(path: Path, block_bytes: int = 16 * 1024 * 1024
                            ) -> tuple[str, os.stat_result]:
    hashes, stat = file_hash_observation(path, block_bytes)
    return hashes["sha256"], stat


def sha256_file(path: Path, block_bytes: int = 16 * 1024 * 1024) -> str:
    """Hash a file and reject a concurrent modification."""
    return sha256_file_observation(path, block_bytes)[0]


def git_blob_sha1(path: Path) -> str:
    """Return Git's blob object ID for a small regular file."""
    return file_hash_observation(path, include_git_blob=True)[0]["git_blob_sha1"]


def shard_paths(checkpoint: Path, nproc: int) -> list[Path]:
    expected = [checkpoint / f"model{rank}-mp{nproc}.safetensors" for rank in range(nproc)]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        fail("missing expected shard(s): " + ", ".join(missing))
    discovered = sorted(checkpoint.glob(f"model*-mp{nproc}.safetensors"))
    if discovered != sorted(expected):
        fail(f"expected exactly {nproc} mp{nproc} shards, found {len(discovered)}")
    return expected


def source_revision_evidence(snapshot: Path, revision: str,
                             required_only: bool = False) -> dict[str, Any]:
    """Validate Hugging Face local-dir cache metadata for an exact commit.

    The metadata format is ``commit_hash``, ``etag``, and download timestamp on
    three lines.  Also reject source files changed after that timestamp; otherwise
    a stale metadata file could falsely attest edited conversion code.
    """
    metadata_root = snapshot / ".cache" / "huggingface" / "download"
    metadata_files = sorted(metadata_root.rglob("*.metadata")) if metadata_root.is_dir() else []
    if required_only:
        wanted = {f"{relative}.metadata" for relative in REQUIRED_SOURCE_FILES}
        metadata_files = [
            path for path in metadata_files
            if path.relative_to(metadata_root).as_posix() in wanted
        ]
        found = {path.relative_to(metadata_root).as_posix() for path in metadata_files}
        if found != wanted:
            missing = sorted(wanted - found)
            fail("required source metadata is missing: " + ", ".join(missing))
    if not metadata_files:
        fail(
            f"no Hugging Face local-dir metadata found under {metadata_root}; "
            "download with `hf download --revision <commit> --local-dir ...`"
        )
    commits: set[str] = set()
    attested_files = []
    required_hashes: dict[str, str] = {}
    for path in metadata_files:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            commit = lines[0].strip()
            etag = lines[1].strip().strip('"')
            timestamp = float(lines[2].strip())
        except (OSError, UnicodeError, IndexError, ValueError) as exc:
            fail(f"cannot read revision metadata {path}: {exc}")
        if not REVISION_RE.fullmatch(commit):
            fail(f"invalid commit hash in revision metadata {path}: {commit!r}")
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", etag):
            fail(f"unsupported ETag in revision metadata {path}: {etag!r}")
        if not math.isfinite(timestamp) or timestamp < 0:
            fail(f"invalid download timestamp in revision metadata {path}")
        commits.add(commit)
        relative = path.relative_to(metadata_root).as_posix()
        if not relative.endswith(".metadata"):
            fail(f"unexpected Hugging Face metadata path: {path}")
        downloaded = snapshot / relative[:-len(".metadata")]
        if not downloaded.is_file():
            fail(f"download metadata has no corresponding source file: {downloaded}")
        downloaded_relative = relative[:-len(".metadata")]
        if downloaded_relative in REQUIRED_SOURCE_FILES:
            hashes, downloaded_stat = file_hash_observation(
                downloaded, include_git_blob=True)
            actual_etag = (hashes["git_blob_sha1"] if len(etag) == 40
                           else hashes["sha256"])
            if actual_etag != etag:
                fail(f"source content does not match Hugging Face ETag: {downloaded}")
            required_hashes[downloaded_relative] = hashes["sha256"]
        else:
            downloaded_stat = downloaded.stat()
        # Match huggingface_hub's one-second tolerance for coarse mtimes.
        if downloaded_stat.st_mtime - 1 > timestamp:
            fail(f"source snapshot file changed after download: {downloaded}")
        attested_files.append(downloaded_relative)
    if commits != {revision}:
        fail(
            "snapshot cache metadata is not uniformly pinned to the requested "
            f"revision {revision}: found {sorted(commits) or 'no commit hashes'}"
        )
    return {
        "metadata_root": str(metadata_root.resolve()),
        "metadata_files": len(metadata_files),
        "attested_files": attested_files,
        "commit_hashes": sorted(commits),
        "required_source_files_sha256": required_hashes,
    }


def required_source_files(snapshot: Path) -> dict[str, str]:
    return {relative: sha256_file(snapshot / relative)
            for relative in REQUIRED_SOURCE_FILES}


def checkpoint_content_identity(record: dict[str, Any]) -> str:
    """Canonical model-byte identity, excluding paths, mtimes, and creation time."""
    model = require_dict(record.get("model"), "model")
    conversion = require_dict(record.get("conversion"), "conversion")
    checkpoint = require_dict(record.get("checkpoint"), "checkpoint")
    source_hashes = require_dict(conversion.get("source_files_sha256"),
                                 "conversion.source_files_sha256")
    shards = checkpoint.get("shards")
    if not isinstance(shards, list):
        fail("checkpoint.shards must be a list")
    canonical_shards = []
    for item in shards:
        item = require_dict(item, "checkpoint shard")
        canonical_shards.append({key: item.get(key) for key in ("name", "size", "sha256")})
    payload = {
        "identity_schema_version": 1,
        "model": {key: model.get(key) for key in ("id", "revision", "inference_revision")},
        "conversion": {
            "n_experts": conversion.get("n_experts"),
            "model_parallel": conversion.get("model_parallel"),
            "expert_dtype": conversion.get("expert_dtype"),
            "source_files_sha256": dict(sorted(source_hashes.items())),
        },
        "checkpoint": {
            "nproc": checkpoint.get("nproc"),
            "shards": sorted(canonical_shards, key=lambda item: str(item["name"])),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, record: dict[str, Any], force: bool) -> None:
    """Durably publish a sidecar without exposing a partially written JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        fail(f"refusing to replace a symlink: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                           dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(record, fp, indent=2, sort_keys=True)
            fp.write("\n")
            fp.flush()
            os.fsync(fp.fileno())
        if force:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                fail(f"output already exists: {path} (pass --force to replace it)")
            temporary.unlink()
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass  # Some network filesystems do not support directory fsync.
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_revision(revision: str, label: str) -> None:
    if not REVISION_RE.fullmatch(revision):
        fail(f"{label} must be an exact 40-character lowercase commit hash")


def validate_pinned_inputs(args: argparse.Namespace) -> None:
    validate_revision(args.model_revision, "--model-revision")
    validate_revision(args.inference_revision, "--inference-revision")
    if (args.model_id, args.model_revision, args.inference_revision) != (
            PINNED_MODEL_ID, PINNED_REVISION, PINNED_REVISION):
        fail(
            "this implementation is validated only for "
            f"{PINNED_MODEL_ID}@{PINNED_REVISION}"
        )
    config = Path(args.config).resolve()
    if sha256_file(config) != PINNED_CONFIG_SHA256:
        fail(f"configuration is not the exact pinned V4-Flash config: {config}")


def create(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).resolve()
    snapshot = Path(args.source_snapshot).resolve()
    output = checkpoint / DEFAULT_NAME
    validate_pinned_inputs(args)
    if not checkpoint.is_dir():
        fail(f"checkpoint directory not found: {checkpoint}")
    if not snapshot.is_dir():
        fail(f"source snapshot directory not found: {snapshot}")
    if output.exists() and not args.force:
        fail(f"output already exists: {output} (pass --force to replace it)")
    if output.is_symlink():
        fail(f"refusing to replace a symlink: {output}")

    evidence = source_revision_evidence(snapshot, args.model_revision)
    sources = dict(evidence["required_source_files_sha256"])
    if not set(sources).issubset(evidence["attested_files"]):
        missing = sorted(set(sources) - set(evidence["attested_files"]))
        fail("required conversion source files lack pinned download metadata: "
             + ", ".join(missing))
    if sources["inference/config.json"] != PINNED_CONFIG_SHA256:
        fail(
            "source snapshot inference/config.json differs from the checked-in "
            "configuration for the pinned V4-Flash revision"
        )
    shards = []
    observed_shards = []
    for index, path in enumerate(shard_paths(checkpoint, args.nproc), 1):
        print(f"hashing shard {index}/{args.nproc}: {path.name}", flush=True)
        digest, stat = sha256_file_observation(path)
        shards.append({
            "name": path.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest,
        })
        observed_shards.append((path, (
            stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
        )))

    # Catch replacements between an individual full read and publication of the
    # multi-shard record. The digest and recorded metadata above still come from
    # one opened inode; this final pass ensures those inodes remain at their paths.
    for path, observed in observed_shards:
        current = path.stat()
        if (current.st_dev, current.st_ino, current.st_size,
                current.st_mtime_ns) != observed:
            fail(f"checkpoint shard changed before provenance publication: {path}")
    current_sources = source_revision_evidence(
        snapshot, args.model_revision, required_only=True)
    if current_sources["required_source_files_sha256"] != sources:
        fail("required source files changed before provenance publication")

    record = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {
            "id": args.model_id,
            "revision": args.model_revision,
            "inference_revision": args.inference_revision,
        },
        "conversion": {
            "source_snapshot": str(snapshot),
            "source_revision_evidence": evidence,
            "source_files_sha256": sources,
            "n_experts": 256,
            "model_parallel": args.nproc,
            "expert_dtype": "fp4",
        },
        "checkpoint": {
            "root_at_creation": str(checkpoint),
            "nproc": args.nproc,
            "shards": shards,
        },
    }
    record["content_identity_sha256"] = checkpoint_content_identity(record)
    write_json_atomic(output, record, args.force)
    print(f"wrote {output}")


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    return value


def verify(args: argparse.Namespace) -> None:
    validate_pinned_inputs(args)
    checkpoint = Path(args.checkpoint).resolve()
    manifest = Path(args.manifest).resolve() if args.manifest else checkpoint / DEFAULT_NAME
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read {manifest}: {exc}")
    record = require_dict(record, "manifest")
    if record.get("schema_version") != SCHEMA_VERSION:
        fail(f"unsupported schema version in {manifest}")
    recorded_content_identity = record.get("content_identity_sha256")
    if (not isinstance(recorded_content_identity, str)
            or not SHA256_RE.fullmatch(recorded_content_identity)
            or recorded_content_identity != checkpoint_content_identity(record)):
        fail("checkpoint content identity is missing or inconsistent")

    model = require_dict(record.get("model"), "model")
    expected_model = {
        "id": args.model_id,
        "revision": args.model_revision,
        "inference_revision": args.inference_revision,
    }
    if model != expected_model:
        fail(f"model declaration mismatch: recorded {model!r}, expected {expected_model!r}")

    conversion = require_dict(record.get("conversion"), "conversion")
    evidence = require_dict(conversion.get("source_revision_evidence"),
                            "conversion.source_revision_evidence")
    if evidence.get("commit_hashes") != [args.model_revision]:
        fail("source snapshot revision evidence does not match --model-revision")
    attested_files = evidence.get("attested_files")
    if (not isinstance(attested_files, list) or not attested_files
            or any(not isinstance(path, str) or not path for path in attested_files)
            or len(attested_files) != len(set(attested_files))
            or evidence.get("metadata_files") != len(attested_files)):
        fail("source snapshot attested-file evidence is malformed")
    source_hashes = require_dict(conversion.get("source_files_sha256"),
                                 "conversion.source_files_sha256")
    required_sources = set(REQUIRED_SOURCE_FILES)
    if set(source_hashes) != required_sources or any(
            not isinstance(value, str) or not SHA256_RE.fullmatch(value)
            for value in source_hashes.values()):
        fail("source-file hashes are missing or malformed")
    if not required_sources.issubset(attested_files):
        fail("required conversion files are not covered by pinned snapshot metadata")
    if evidence.get("required_source_files_sha256") != source_hashes:
        fail("source revision evidence and recorded source hashes disagree")
    if args.source_snapshot:
        snapshot = Path(args.source_snapshot).resolve()
        current_evidence = source_revision_evidence(
            snapshot, args.model_revision, required_only=True)
        if not required_sources.issubset(current_evidence["attested_files"]):
            fail("current source snapshot lacks pinned metadata for required files")
        if current_evidence["required_source_files_sha256"] != source_hashes:
            fail("current inference/encoding source differs from the conversion snapshot")
    if (conversion.get("n_experts"), conversion.get("model_parallel"),
            conversion.get("expert_dtype")) != (256, args.nproc, "fp4"):
        fail("conversion parameters do not match the required V4 mp configuration")

    checkpoint_record = require_dict(record.get("checkpoint"), "checkpoint")
    if checkpoint_record.get("nproc") != args.nproc:
        fail("checkpoint nproc does not match the requested topology")
    stored = checkpoint_record.get("shards")
    if not isinstance(stored, list) or len(stored) != args.nproc:
        fail("checkpoint shard record has the wrong length")
    by_name = {}
    for item in stored:
        item = require_dict(item, "checkpoint shard")
        name, size, mtime_ns, digest = (
            item.get("name"), item.get("size"), item.get("mtime_ns"), item.get("sha256")
        )
        if (not isinstance(name, str) or isinstance(size, bool) or not isinstance(size, int)
                or size < 0 or isinstance(mtime_ns, bool) or not isinstance(mtime_ns, int)
                or not isinstance(digest, str) or not SHA256_RE.fullmatch(digest)):
            fail("checkpoint shard record is malformed")
        if name in by_name:
            fail(f"duplicate checkpoint shard record: {name}")
        by_name[name] = item

    for path in shard_paths(checkpoint, args.nproc):
        item = by_name.get(path.name)
        if item is None:
            fail(f"missing checkpoint shard record: {path.name}")
        stat = path.stat()
        if stat.st_size != item["size"]:
            fail(
                f"checkpoint shard size changed since hashing: {path.name}; "
                "recreate the provenance record after verifying the checkpoint"
            )
        if stat.st_mtime_ns != item["mtime_ns"] and not args.full:
            fail(
                f"checkpoint shard mtime changed since hashing: {path.name}; "
                "run verify --full or recreate the provenance record"
            )
        if stat.st_mtime_ns != item["mtime_ns"] and args.full:
            print(f"warning: {path.name} mtime changed; verifying content", flush=True)
        if args.full:
            print(f"rehashing {path.name}", flush=True)
            if sha256_file(path) != item["sha256"]:
                fail(f"checkpoint shard content hash mismatch: {path.name}")

    print(f"verified {manifest} ({'full hashes' if args.full else 'metadata + recorded hashes'})")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", required=True)
    common.add_argument("--model-id", required=True)
    common.add_argument("--model-revision", required=True)
    common.add_argument("--inference-revision", required=True)
    common.add_argument("--config", default=str(Path(__file__).with_name("config_v4_flash.json")))
    common.add_argument("--nproc", type=int, default=4)

    create_parser = sub.add_parser("create", parents=[common])
    create_parser.add_argument("--source-snapshot", required=True)
    create_parser.add_argument("--force", action="store_true")

    verify_parser = sub.add_parser("verify", parents=[common])
    verify_parser.add_argument("--manifest", default="")
    verify_parser.add_argument("--source-snapshot", default="",
                               help="also compare the current inference/encoding snapshot")
    verify_parser.add_argument("--full", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if isinstance(args.nproc, bool) or args.nproc < 1:
        fail("--nproc must be a positive integer")
    {"create": create, "verify": verify}[args.mode](args)


if __name__ == "__main__":
    main()
