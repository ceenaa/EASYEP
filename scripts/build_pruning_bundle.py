#!/usr/bin/env python3
"""Package tracked FreeToken source and this adapter, excluding weights/data/environments."""

import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT / "FreeToken").decode().split("\0")
    paths = {ROOT / "FreeToken" / name for name in tracked if name}
    for folder in ("FreeToken/python/freetoken/pruning", "FreeToken/tests/pruning"):
        paths.update((ROOT / folder).glob("*.py"))
    paths.update(p for p in (ROOT / "launcher").glob("*") if p.is_file())
    paths.update(ROOT / name for name in (
        "RUN_ON_VAST.md", "SECURITY_EXPERIMENT.md", "RESEARCH.md", "VALIDATION.md",
        "PRIMEVUL_EVALUATION.md", "PR1_COMPARISON.md",
        "scripts/start_pruning_server.py", "scripts/inspect_target.py", "scripts/build_pruning_bundle.py",
        "scripts/evaluate_full_model.py", "scripts/run_full_baseline.sh", "v03_extended.txt",
    ))
    paths = sorted(path for path in paths if path.is_file())
    destination = ROOT / "dist"
    destination.mkdir(exist_ok=True)
    package = destination / "deepseek-v4-easyep-ready.tar.gz"
    with tarfile.open(package, "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=Path("prunungdsk4") / path.relative_to(ROOT), recursive=False)
    manifest = {"archive": package.name, "sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
                "base_freetoken_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / "FreeToken").decode().strip(),
                "files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (destination / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"archive": str(package), "bytes": package.stat().st_size, "files": len(paths), "sha256": manifest["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
