#!/usr/bin/env python3
"""Materialise PrimeVul into the corpus contract easyep_v4.py already consumes.

PrimeVul (ICSE 2025, arXiv:2403.18624, MIT) ships JSONL rows carrying C/C++
function source *inline*. The V4 pipeline consumes something different: a
directory of source files plus a matched-pair manifest that binds every member
by SHA-256, because that is what lets calibration sampling, matched-pair
selection and the run manifest hash the same bytes the model actually saw.

Converting into that contract -- rather than adding a second loader beside it --
is deliberate. Everything that makes the pipeline auditable lives in the
consumers: balanced vulnerable/secure calibration sampling that never takes both
sides of one pair, path-level exclusion of calibrated files from the evaluation
set, content hashes in the run manifest, and the fail-closed checks on all of
it. A parallel PrimeVul loader would have to re-earn every one of those. A
converter earns them once, at the cost of writing the corpus to disk.

What PrimeVul's labels mean, and how that differs from the CodeQL corpus
--------------------------------------------------------------------
The existing `vulnerable-js-files` corpus pairs a file CodeQL flagged with the
same file after the flagged expression was neutralised and the copy rescanned
clean. Ground truth is "a static analyser's alert, and its absence".

A PrimeVul pair is the pre- and post-fix revision of one function from a real
vulnerability-fixing commit. Ground truth is "maintainers shipped a fix here".
That is a stronger claim about the vulnerable member and a *weaker* one about
the benign member: the post-fix function is only known not to contain the
vulnerability that commit fixed. It is not certified free of every other one,
the way a clean CodeQL rescan certifies the absence of that scanner's alerts.

The matched-pair metric asks the model for a binary VULNERABLE/SAFE verdict on
each member, so this matters for reading the result: a PrimeVul "SAFE" member
that the model calls VULNERABLE may be a genuine second bug rather than a false
positive, and the metric cannot tell those apart. The two corpora therefore
measure related but non-identical things, which is why easyep_v4.py records a
corpus identity in every score artifact and refuses to evaluate pairs from one
corpus against masks calibrated on the other.

Splits
------
Multiple PrimeVul splits can be materialised into one corpus root and every
manifest row records which split it came from, which `easyep_v4.py` selects on
via `--calib-splits` / `--pairs-splits`. Convert all three paired splits into
*one* root: corpus identity is the manifest digest, so two roots would make
calibration and matched-pair evaluation refuse each other.

Recommended is `--calib-splits train_paired` with `--pairs-splits test_paired`,
which keeps the whole 433-pair test set available instead of spending part of it
on profiling. Naming no split draws from every split, which on a multi-split
root is rarely intended.

Split separation is hygiene rather than necessity: EASY-EP calibration is
unsupervised -- it profiles which experts route, it fits nothing to labels -- so
no trained parameter can memorise a test item, and path-level disjointness is
what carries the claim. Splitting just makes the result easy to defend.

Usage
-----
    python v4/primevul_dataset.py fetch          # how to obtain the data
    python v4/primevul_dataset.py describe --primevul-dir DIR
    python v4/primevul_dataset.py convert --primevul-dir DIR \\
        --out $DATA/primevul-c-files --split test_paired
    python v4/primevul_dataset.py verify --root $DATA/primevul-c-files

The corpus root produced here is a drop-in `--calib-dir` and its
PRIMEVUL_PAIRED_MANIFEST.jsonl a drop-in `--pairs-manifest`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import os
import shutil
import sys
from pathlib import Path

CONVERTER_VERSION = "primevul-corpus-v1"
MANIFEST_NAME = "PRIMEVUL_PAIRED_MANIFEST.jsonl"
PROVENANCE_NAME = "DATASET_PROVENANCE.json"
GROUND_TRUTH = "primevul_commit_pair"
METHOD = "primevul_vulnerability_fixing_commit"

# PrimeVul's own file naming. Only paired splits are usable: the corpus contract
# is a matched-pair manifest, and an unpaired split has no benign counterpart to
# pair a vulnerable function with.
PAIRED_SPLITS = ("train_paired", "valid_paired", "test_paired")

# The label field is `target` in the release the paper used. Mirrors and the
# v0.1 re-release have been seen carrying `label`, so a small explicit alias set
# is accepted -- and which one was used is recorded, because "we guessed the
# label column" is exactly the kind of thing that must not be invisible later.
LABEL_FIELDS = ("target", "label")
CODE_FIELDS = ("func", "func_before")

DOWNLOAD_HELP = """PrimeVul is distributed through Google Drive, so it cannot be fetched
non-interactively from here. To obtain it:

  1. Open https://github.com/DLVulDet/PrimeVul and follow the Google Drive link
     in the README ("Latest Release (v0.1)" carries CVE/commit metadata; the
     "Original Release" is the exact set used in the ICSE 2025 paper).
  2. Download at least one *paired* split -- primevul_test_paired.jsonl is the
     435-pair set the paper itself uses for paired evaluation.
  3. Put the .jsonl files in one directory, then run:

       python v4/primevul_dataset.py describe --primevul-dir <that directory>
       python v4/primevul_dataset.py convert  --primevul-dir <that directory> \\
           --out "$EASYEP_DATA_ROOT/primevul-c-files" --split test_paired

PrimeVul is MIT licensed. Like deepseek_easy_ep_inputs/, the converted corpus is
experiment data: keep it out of git and let the run manifest record its digest.
"""

# Unambiguous C++ markers. The extension only reaches the model as the neutral
# display path's suffix (review_target.c / .cpp), so a wrong guess is a wrong
# language hint rather than a leak -- but it is decided once per pair, from the
# vulnerable member, so both members always show the identical path.
CPP_MARKERS = ("::", "template<", "template <", "namespace ", "nullptr",
               "public:", "private:", "protected:", "virtual ", "throw ",
               "catch (", "catch(", "new (")

# Suffixes PrimeVul's own file_name is allowed to contribute. Anything else
# (a .S, a .py test harness) falls back to the heuristic rather than showing the
# model a language the corpus does not claim to be.
SOURCE_SUFFIXES = frozenset({".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh"})

_CWE_RE = re.compile(r"CWE-\d+", re.IGNORECASE)
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


# ------------------------------------------------------------------ reading

def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _first_field(row: dict, names: tuple[str, ...]) -> tuple[str, object] | None:
    for name in names:
        if name in row:
            return name, row[name]
    return None


def read_primevul_rows(path: Path) -> tuple[list[dict], dict]:
    """Parse one PrimeVul JSONL split into normalised rows.

    Returns the rows plus a small report of which schema variant was found, so
    the caller can record it instead of the fact being lost in a guess.
    """
    text = path.read_text(encoding="utf-8")
    rows: list[dict] = []
    label_field = code_field = None
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{path}:{number} is {type(raw).__name__}, expected an object")

        found_label = _first_field(raw, LABEL_FIELDS)
        found_code = _first_field(raw, CODE_FIELDS)
        if found_label is None:
            raise ValueError(
                f"{path}:{number} has no label field (looked for "
                f"{'/'.join(LABEL_FIELDS)}; keys: {sorted(raw)[:8]})")
        if found_code is None:
            raise ValueError(
                f"{path}:{number} has no source field (looked for "
                f"{'/'.join(CODE_FIELDS)}; keys: {sorted(raw)[:8]})")
        # One split must not mix schemas; that would mean two different releases
        # were concatenated, and the pairing below trusts row order.
        for seen, name, kind in ((label_field, found_label[0], "label"),
                                 (code_field, found_code[0], "source")):
            if seen is not None and seen != name:
                raise ValueError(
                    f"{path}:{number} uses {kind} field {name!r} but earlier rows "
                    f"used {seen!r}; this file mixes PrimeVul schema variants")
        label_field, code_field = found_label[0], found_code[0]

        target = found_label[1]
        if isinstance(target, bool) or target not in (0, 1):
            raise ValueError(
                f"{path}:{number} has {label_field}={target!r}; PrimeVul labels "
                "are 1 (vulnerable) or 0 (benign)")
        func = found_code[1]
        if not isinstance(func, str) or not func.strip():
            raise ValueError(f"{path}:{number} has empty {code_field}")

        rows.append({
            "line": number,
            "target": int(target),
            "func": func,
            "commit_id": str(raw.get("commit_id") or "").strip(),
            "project": str(raw.get("project") or "").strip(),
            "cwe": normalise_cwe(raw),
            "cve": str(raw.get("cve") or "").strip(),
            # v0.1 serialises a missing filename as the *string* "None" (20-36%
            # of rows, measured across the three paired splits), so a truthiness
            # test alone would hand that through as a real name.
            "file_name": ("" if str(raw.get("file_name") or "").strip() in ("", "None", "none")
                          else str(raw["file_name"]).strip()),
            "idx": raw.get("idx"),
        })
    if not rows:
        raise ValueError(f"{path} contains no records")
    return rows, {"label_field": label_field, "code_field": code_field,
                  "records": len(rows), "sha256": _sha256_text(text)}


def normalise_cwe(row: dict) -> list[str]:
    """Every CWE id mentioned by the record, uppercased and de-duplicated.

    The original release stores this as a string that is often empty; v0.1
    stores a list. Both reach here.
    """
    # `or`, not a get-default: the original release ships `"cwe": ""`, which is
    # present but empty, and would otherwise shadow a populated alias and drop
    # every pair into CWE-UNKNOWN -- collapsing the strata calibration samples on.
    value = row.get("cwe") or row.get("cwe_ids") or ""
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple)):
        candidates = [str(item) for item in value]
    else:
        candidates = []
    out: list[str] = []
    for candidate in candidates:
        for match in _CWE_RE.findall(candidate):
            token = match.upper()
            if token not in out:
                out.append(token)
    return out


def stratum_of(cwes: list[str]) -> str:
    """The directory the pair lives under, which is what sampling stratifies on.

    Unknown-CWE pairs get their own stratum rather than being dropped: over half
    of the original release carries no CWE, and discarding them would silently
    narrow the corpus to whatever subset happened to be annotated.
    """
    return cwes[0] if cwes else "CWE-UNKNOWN"


def guess_extension(func: str, file_name: str = "") -> str:
    """The suffix the pair is shown under, preferring PrimeVul's own filename.

    v0.1 records the originating file, which settles .c/.cc/.cpp/.h exactly; the
    marker heuristic is only the fallback for the fifth to third of rows that
    carry no name. Decided once per pair from the vulnerable member, so both
    members always display the identical neutral path.
    """
    if file_name:
        suffix = Path(file_name).suffix.lower()
        if suffix in SOURCE_SUFFIXES:
            return suffix
    return ".cpp" if any(marker in func for marker in CPP_MARKERS) else ".c"


# ------------------------------------------------------------------ pairing

def pair_rows(rows: list[dict], source: str,
              stats: dict | None = None) -> list[tuple[dict, dict]]:
    """Adjacent rows form (vulnerable, benign) pairs -- verified, not assumed.

    PrimeVul's paired splits list the two revisions of a function consecutively.
    Nothing in the file marks a pair boundary, so a single misaligned row would
    silently pair every later function with the wrong counterpart and produce a
    corpus whose labels are shifted by one. Each pair is therefore checked, and
    a violation aborts with line numbers rather than being repaired by guesswork.

    What is checked is what actually holds. Measured over all three paired splits
    of the v0.1 release (870 + 7578 + 960 rows): every adjacent pair carries one
    target=1 and one target=0, every row names a commit, and no pair has
    identical members -- so those are hard errors. But the two members carry
    *different* commit ids in 19 pairs and different file names in 13 more
    (with either name absent in 1635), because PrimeVul sometimes draws the
    benign revision from a later commit than the one that introduced the flaw.
    Those are real pairs, so treating a mismatch as corruption would discard
    them; they are counted into `stats` and reported instead, which keeps the
    fact visible without inventing a rule the dataset does not follow.
    """
    if len(rows) % 2:
        raise ValueError(
            f"{source} has {len(rows)} records; a paired split must have an even "
            "count, two rows per function")
    pairs = []
    anomalies = {"commit_id_differs": 0, "file_name_differs": 0, "file_name_absent": 0}
    for first, second in zip(rows[::2], rows[1::2]):
        where = f"{source}:{first['line']},{second['line']}"
        if {first["target"], second["target"]} != {0, 1}:
            raise ValueError(
                f"{where} both have target={first['target']}; adjacent rows in a "
                "paired split must be one vulnerable and one benign. Is this an "
                "unpaired split?")
        vulnerable = first if first["target"] == 1 else second
        benign = second if first["target"] == 1 else first
        if not vulnerable["commit_id"]:
            raise ValueError(f"{where} has no commit_id; the label has no provenance")
        if vulnerable["func"] == benign["func"]:
            raise ValueError(
                f"{where} have identical source, so the fix is not represented "
                "and the pair cannot discriminate anything")
        if first["commit_id"] != second["commit_id"]:
            anomalies["commit_id_differs"] += 1
        if not (first["file_name"] and second["file_name"]):
            anomalies["file_name_absent"] += 1
        elif first["file_name"] != second["file_name"]:
            anomalies["file_name_differs"] += 1
        pairs.append((vulnerable, benign))
    if stats is not None:
        stats.update(anomalies)
    return pairs


# --------------------------------------------------------------- conversion

def _safe(name: str, fallback: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("-", name).strip("-.")
    return cleaned[:48] if cleaned else fallback


def convert(primevul_dir: Path, out: Path, splits: list[str], *,
            max_pairs: int = 0, max_func_chars: int = 0,
            overwrite: bool = False) -> dict:
    """Write the corpus tree, the matched-pair manifest and its provenance."""
    primevul_dir = primevul_dir.resolve()
    out = out.resolve()
    if not primevul_dir.is_dir():
        raise SystemExit(f"not a directory: {primevul_dir}\n\n{DOWNLOAD_HELP}")
    if out.exists():
        if not overwrite:
            raise SystemExit(
                f"{out} already exists. Converting again would leave a mixture of "
                "two conversions whose manifest describes only the newer one; "
                "remove it or pass --overwrite.")
        if not (out / MANIFEST_NAME).is_file():
            raise SystemExit(
                f"--overwrite refused: {out} exists but carries no {MANIFEST_NAME}, "
                "so it is not a corpus this converter produced.")

    sources, all_pairs = [], []
    for split in splits:
        path = primevul_dir / f"primevul_{split}.jsonl"
        if not path.is_file():
            available = sorted(p.name for p in primevul_dir.glob("primevul_*.jsonl"))
            raise SystemExit(
                f"missing {path.name} in {primevul_dir}\n"
                f"found: {', '.join(available) if available else '(none)'}\n\n"
                f"{DOWNLOAD_HELP}")
        rows, report = read_primevul_rows(path)
        anomalies: dict = {}
        pairs = pair_rows(rows, path.name, stats=anomalies)
        report.update({"file": path.name, "split": split, "pairs": len(pairs),
                       "pair_anomalies": anomalies})
        sources.append(report)
        all_pairs.extend((split, vulnerable, benign) for vulnerable, benign in pairs)

    stats = {"pairs_read": len(all_pairs), "skipped_duplicate": 0,
             "skipped_oversized": 0, "written": 0}

    # Build into a sibling and swap only once the manifest is written.
    #
    # Two failure modes force this. Writing into a populated root would *merge*:
    # every pair the new conversion does not reproduce stays on disk as an
    # orphan the manifest does not list, invisible to verify() but still counted
    # in the data-root digest the run manifest records. And rmtree-ing the root
    # up front would mean any later abort -- an unmet --max-pairs, a full disk --
    # destroys an existing good corpus and leaves nothing. Staging gives
    # replace-not-merge without ever putting the old corpus at risk.
    staging = out.with_name(f"{out.name}.converting-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        provenance = _convert_pairs(staging, all_pairs, stats, max_pairs,
                                    max_func_chars, splits, sources, out.name)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    previous = out.with_name(f"{out.name}.replaced-{os.getpid()}")
    if out.exists():
        out.rename(previous)
    try:
        staging.rename(out)
    except BaseException:
        if previous.exists():
            previous.rename(out)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(previous, ignore_errors=True)
    return provenance


def _convert_pairs(out: Path, all_pairs: list, stats: dict, max_pairs: int,
                   max_func_chars: int, splits: list[str], sources: list[dict],
                   corpus_name: str) -> dict:
    """Write the tree and manifest under `out`, naming paths for `corpus_name`.

    Paths are recorded relative to the corpus root's *parent*, so they must
    carry the final directory name rather than the staging one.
    """
    seen: set[tuple[str, str]] = set()
    seen_single: set[str] = set()
    manifest_rows, sizes, languages = [], [], {}
    for split, vulnerable, benign in all_pairs:
        if max_pairs and stats["written"] >= max_pairs:
            break
        vuln_sha = _sha256_text(vulnerable["func"])
        safe_sha = _sha256_text(benign["func"])
        # PrimeVul de-duplicates within a split; identical functions still recur
        # across splits, and one function appearing under two labels would let a
        # calibration draw and an evaluation item share bytes while the
        # path-based exclusion sees two different files.
        if (vuln_sha, safe_sha) in seen or vuln_sha in seen_single or safe_sha in seen_single:
            stats["skipped_duplicate"] += 1
            continue
        if max_func_chars and (len(vulnerable["func"]) > max_func_chars
                               or len(benign["func"]) > max_func_chars):
            stats["skipped_oversized"] += 1
            continue

        cwes = vulnerable["cwe"] or benign["cwe"]
        stratum = stratum_of(cwes)
        extension = guess_extension(vulnerable["func"], vulnerable["file_name"])
        languages[extension] = languages.get(extension, 0) + 1
        commit = vulnerable["commit_id"]
        leaf = "__".join((
            _safe(vulnerable["project"], "project"),
            _safe(commit[:12], "commit"),
            _safe(str(vulnerable.get("idx") if vulnerable.get("idx") is not None
                      else vulnerable["line"]), str(vulnerable["line"])),
        ))
        directory = out / stratum / leaf
        if directory.exists():
            leaf = f"{leaf}-{len(manifest_rows)}"
            directory = out / stratum / leaf
        directory.mkdir(parents=True)

        # Same stem plus the corpus's secure-copy suffix, mirroring the CodeQL
        # tree. neutral_display_path strips both before the model sees anything.
        vuln_path = directory / f"func{extension}"
        safe_path = directory / f"func_Code{extension}"
        vuln_path.write_text(vulnerable["func"], encoding="utf-8")
        safe_path.write_text(benign["func"], encoding="utf-8")

        manifest_rows.append({
            "original_file": f"{corpus_name}/{vuln_path.relative_to(out).as_posix()}",
            "secure_file": f"{corpus_name}/{safe_path.relative_to(out).as_posix()}",
            "extension": extension,
            "original_sha256": vuln_sha,
            "secure_sha256": safe_sha,
            # What the labels are evidence of. easyep_v4.pair_row_is_labelled
            # admits this row on the commit, not on a CodeQL alert count -- see
            # this module's docstring for why the two are not interchangeable.
            "ground_truth": GROUND_TRUTH,
            "method": METHOD,
            "commit_id": commit,
            "project": vulnerable["project"],
            "file_name": vulnerable["file_name"],
            "cwe": cwes,
            "cve": vulnerable["cve"] or benign["cve"],
            # load_pairs surfaces `queries` in its results; the CWE list is the
            # nearest PrimeVul analogue of "what this pair is about".
            "queries": cwes,
            "primevul_split": split,
            "primevul_idx": [vulnerable.get("idx"), benign.get("idx")],
        })
        seen.add((vuln_sha, safe_sha))
        seen_single.update((vuln_sha, safe_sha))
        sizes.extend((len(vulnerable["func"]), len(benign["func"])))
        stats["written"] += 1

    if not manifest_rows:
        raise SystemExit("no usable pairs survived conversion; see the stats above")
    if max_pairs and stats["written"] < max_pairs:
        raise SystemExit(
            f"requested {max_pairs} pairs but only {stats['written']} were usable "
            f"({stats['skipped_duplicate']} duplicate, "
            f"{stats['skipped_oversized']} oversized)")

    manifest = out / MANIFEST_NAME
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8")

    provenance = {
        "converter": CONVERTER_VERSION,
        "dataset": "PrimeVul",
        "dataset_url": "https://github.com/DLVulDet/PrimeVul",
        "dataset_paper": "arXiv:2403.18624",
        "ground_truth": GROUND_TRUTH,
        "splits": list(splits),
        "sources": sources,
        "stats": stats,
        "languages": languages,
        "strata": len({row["original_file"].split("/")[1] for row in manifest_rows}),
        "member_characters": _percentiles(sizes),
        "manifest": MANIFEST_NAME,
        # The manifest embeds this name in every path; see verify().
        "corpus_root_name": corpus_name,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "parameters": {"max_pairs": max_pairs, "max_func_chars": max_func_chars},
    }
    (out / PROVENANCE_NAME).write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    return provenance


def _percentiles(values: list[int]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def at(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {"n": len(ordered), "min": ordered[0], "median": at(0.5),
            "p90": at(0.9), "p99": at(0.99), "max": ordered[-1]}


# ------------------------------------------------------------ verification

def verify(root: Path) -> dict:
    """Re-check a converted corpus against its manifest.

    The run manifest hashes the data root, but that only proves the tree has not
    changed since a run started. This proves the tree still matches the pairing
    the converter recorded, which is what the labels rest on.
    """
    root = root.resolve()
    manifest = root / MANIFEST_NAME
    if not manifest.is_file():
        raise SystemExit(f"no {MANIFEST_NAME} under {root}")
    rows = [json.loads(line) for line in
            manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{manifest} has no rows")
    # Every consumer -- load_pairs, _manifest_calibration_candidates and verify
    # alike -- addresses a member as root.parent / original_file, so the manifest
    # embeds the corpus directory's name and renaming the root silently detaches
    # it from its own manifest. Diagnose that here, where the cause is still
    # visible, rather than letting it surface mid-run as a path-escape abort.
    declared = {row.get("original_file", "").split("/", 1)[0] for row in rows}
    if declared != {root.name}:
        raise SystemExit(
            f"{root} is named {root.name!r} but its manifest addresses members "
            f"under {'/'.join(sorted(declared))!r}. Pair files are resolved "
            "relative to the parent of the corpus root, so the directory must "
            f"keep the name it was converted as; rename it back to "
            f"{sorted(declared)[0]!r} or re-convert.")
    seen_paths: dict[str, int] = {}
    for index, row in enumerate(rows):
        if row.get("ground_truth") != GROUND_TRUTH:
            raise SystemExit(
                f"row {index} declares ground_truth {row.get('ground_truth')!r}, "
                f"expected {GROUND_TRUTH!r}")
        for key, hash_key in (("original_file", "original_sha256"),
                              ("secure_file", "secure_sha256")):
            relative = row[key]
            path = (root.parent / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise SystemExit(f"row {index} {key} escapes the corpus root: {relative}") from exc
            if not path.is_file():
                raise SystemExit(f"row {index} {key} is missing: {relative}")
            if relative in seen_paths:
                raise SystemExit(
                    f"row {index} reuses {relative}, already used by row "
                    f"{seen_paths[relative]}; one file cannot carry two labels")
            seen_paths[relative] = index
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row[hash_key]:
                raise SystemExit(
                    f"row {index} {hash_key} mismatch for {relative}: manifest says "
                    f"{row[hash_key][:12]}, file is {digest[:12]}")
        if row["original_sha256"] == row["secure_sha256"]:
            raise SystemExit(f"row {index} has identical members")
    return {"rows": len(rows), "files": len(seen_paths),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}


# --------------------------------------------------------------------- cli

def _cmd_fetch(_a) -> None:
    print(DOWNLOAD_HELP)


def _cmd_describe(a) -> None:
    directory = Path(a.primevul_dir).resolve()
    found = sorted(directory.glob("primevul_*.jsonl"))
    if not found:
        raise SystemExit(f"no primevul_*.jsonl under {directory}\n\n{DOWNLOAD_HELP}")
    for path in found:
        split = path.stem.replace("primevul_", "")
        try:
            rows, report = read_primevul_rows(path)
        except ValueError as exc:
            print(f"{path.name}: UNREADABLE  {exc}")
            continue
        vulnerable = sum(row["target"] for row in rows)
        line = (f"{path.name}: {report['records']} records "
                f"({vulnerable} vulnerable / {report['records'] - vulnerable} benign), "
                f"label={report['label_field']} source={report['code_field']}")
        if split in PAIRED_SPLITS:
            try:
                anomalies: dict = {}
                pairs = pair_rows(rows, path.name, stats=anomalies)
                line += f", {len(pairs)} pairs"
                noted = ", ".join(f"{key}={value}" for key, value in
                                  sorted(anomalies.items()) if value)
                if noted:
                    line += f" ({noted}; tolerated, see pair_rows)"
            except ValueError as exc:
                line += f", NOT PAIRABLE: {exc}"
        else:
            line += ", unpaired (not usable as a corpus)"
        print(line)


def _cmd_convert(a) -> None:
    for split in a.split:
        if split not in PAIRED_SPLITS:
            raise SystemExit(
                f"{split!r} is not a paired split. The corpus contract is a "
                f"matched-pair manifest, so only {', '.join(PAIRED_SPLITS)} can "
                "be converted.")
    provenance = convert(
        Path(a.primevul_dir), Path(a.out), a.split,
        max_pairs=a.max_pairs, max_func_chars=a.max_func_chars,
        overwrite=a.overwrite)
    stats, sizes = provenance["stats"], provenance["member_characters"]
    print(f"[primevul] wrote {stats['written']} pairs to {a.out}")
    print(f"[primevul]   read {stats['pairs_read']}, skipped "
          f"{stats['skipped_duplicate']} duplicate / "
          f"{stats['skipped_oversized']} oversized")
    print(f"[primevul]   languages {provenance['languages']}, "
          f"{provenance['strata']} CWE strata")
    for report in provenance["sources"]:
        noted = ", ".join(f"{key}={value}" for key, value
                          in sorted(report["pair_anomalies"].items()) if value)
        if noted:
            print(f"[primevul]   {report['file']}: {noted} (tolerated)")
    print(f"[primevul]   member size chars: median {sizes['median']}, "
          f"p90 {sizes['p90']}, p99 {sizes['p99']}, max {sizes['max']}")
    print(f"[primevul]   ~{sizes['p90'] // 4} tokens at p90 (rough 4 chars/token); "
          "pairs over the context budget are rejected at evaluation time")
    print(f"[primevul] manifest {provenance['manifest_sha256'][:16]}")
    print(f"[primevul] use --calib-dir {a.out} "
          f"--pairs-manifest {Path(a.out) / MANIFEST_NAME}")


def _cmd_verify(a) -> None:
    report = verify(Path(a.root))
    print(f"[primevul] OK  {report['rows']} pairs / {report['files']} files, "
          f"manifest {report['manifest_sha256'][:16]}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("fetch", help="print how to obtain PrimeVul").set_defaults(fn=_cmd_fetch)

    describe = sub.add_parser("describe", help="report on downloaded PrimeVul splits")
    describe.add_argument("--primevul-dir", required=True)
    describe.set_defaults(fn=_cmd_describe)

    convert_parser = sub.add_parser("convert", help="materialise a corpus root")
    convert_parser.add_argument("--primevul-dir", required=True)
    convert_parser.add_argument("--out", required=True,
                                help="corpus root to create, e.g. $DATA/primevul-c-files")
    convert_parser.add_argument("--split", action="append", default=None,
                                choices=PAIRED_SPLITS,
                                help="repeatable; default test_paired")
    convert_parser.add_argument("--max-pairs", type=int, default=0,
                                help="stop after N pairs (0 = all)")
    convert_parser.add_argument("--max-func-chars", type=int, default=0,
                                help="skip pairs with a member longer than this "
                                     "(0 = keep all; over-context pairs are "
                                     "rejected at evaluation time regardless)")
    convert_parser.add_argument("--overwrite", action="store_true")
    convert_parser.set_defaults(fn=_cmd_convert)

    verify_parser = sub.add_parser("verify", help="re-check a converted corpus")
    verify_parser.add_argument("--root", required=True)
    verify_parser.set_defaults(fn=_cmd_verify)

    a = parser.parse_args(argv)
    if getattr(a, "cmd", None) == "convert" and not a.split:
        a.split = ["test_paired"]
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
