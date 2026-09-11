"""Import a user-supplied aligned PrimeVul ZIP as held-out source, without inferring labels."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

from .artifacts import write_new

SOURCE_NAME = re.compile(r"sample_(\d+)_pair_(\d+)\.(c|cpp)\Z")
MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 20_000_000


def import_archive(archive_path, output):
    archive_path, output = Path(archive_path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    members, seen_names, seen_samples = [], set(), set()
    pairs = defaultdict(list)
    total = 0
    with zipfile.ZipFile(archive_path) as archive:
        if len(archive.infolist()) > 5000:
            raise ValueError("Too many ZIP members")
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                raise ValueError("Unsafe ZIP member path")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("ZIP symlinks are not supported")
            if info.is_dir() or path.parts[0] == "__MACOSX" or path.name == ".DS_Store":
                continue
            match = SOURCE_NAME.fullmatch(path.name)
            if not match:
                raise ValueError(f"Unexpected member in aligned source ZIP: {info.filename}")
            sample, pair, suffix = match.groups()
            sample, pair = int(sample), int(pair)
            if path.name in seen_names or sample in seen_samples:
                raise ValueError("Duplicate source name or sample ID in ZIP")
            seen_names.add(path.name)
            seen_samples.add(sample)
            if info.file_size > MAX_FILE_BYTES:
                raise ValueError("Source exceeds ZIP import byte limit")
            total += info.file_size
            if total > MAX_TOTAL_BYTES:
                raise ValueError("Sources exceed total ZIP import byte limit")
            with archive.open(info) as handle:
                data = handle.read(MAX_FILE_BYTES + 1)
            if len(data) != info.file_size or not data:
                raise ValueError("Invalid or empty source member")
            data.decode("utf-8")
            source = {"sample": sample, "pair": pair, "suffix": suffix,
                      "name": path.name, "member": info.filename, "data": data,
                      "sha256": hashlib.sha256(data).hexdigest()}
            members.append(source)
            pairs[pair].append(source)
    if not pairs or any(len(group) != 2 or group[0]["suffix"] != group[1]["suffix"] for group in pairs.values()):
        raise ValueError("Expected exactly two same-language source files in every pair")

    archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    rows, hashes = [], defaultdict(list)
    members.sort(key=lambda member: member["sample"])
    for member in members:
        case_id = f"primevul_sample_{member['sample']:03d}"
        rows.append({
            "id": case_id, "pair_id": f"primevul_pair_{member['pair']:02d}",
            "split": "test", "held_out": True,
            "repository": "unknown:primevul-aligned", "repository_status": "unverified",
            "revision": None, "language": "c" if member["suffix"] == "c" else "cpp",
            "files": [f"sources/{member['name']}"],
            "source_sha256": {f"sources/{member['name']}": member["sha256"]},
            "label": None, "expected_findings": [], "ground_truth_status": "missing",
            "archive_member": member["member"], "archive_sha256": archive_hash,
        })
        hashes[member["sha256"]].append(case_id)
    summary = {
        "kind": "primevul_aligned_import", "archive_name": archive_path.name,
        "archive_sha256": archive_hash, "cases": len(rows), "pairs": len(pairs),
        "languages": dict(Counter(row["language"] for row in rows)), "source_bytes": total,
        "split": "test", "ground_truth_status": "missing", "repository_status": "unverified",
        "identical_source_groups": [ids for ids in hashes.values() if len(ids) > 1],
        "label_policy": "No labels inferred from odd/even sample numbers, pair order, or code differences.",
        "separation_limit": "Original repositories are unknown; repository/near-duplicate separation from calibration cannot yet be verified.",
        "source_sha256": {row["files"][0]: next(iter(row["source_sha256"].values())) for row in rows},
    }
    # Validate every member before creating any output; never use extractall on supplied paths.
    output.mkdir(parents=True, exist_ok=False)
    (output / "sources").mkdir()
    for member in members:
        (output / "sources" / member["name"]).write_bytes(member["data"])
    with (output / "manifest.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_new(output / "import-report.json", summary)
    return summary


def import_primevul(args):
    report = import_archive(args.archive, args.output_dir)
    print(json.dumps({key: report[key] for key in (
        "cases", "pairs", "languages", "split", "ground_truth_status", "archive_sha256")}, indent=2))
