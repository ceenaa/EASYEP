import json
import stat
from types import SimpleNamespace
import zipfile

import pytest

from freetoken.pruning.artifacts import read_json, write_new
from freetoken.pruning.primevul import import_archive
from freetoken.pruning import workflow as w


def archive(tmp_path, entries=None):
    path = tmp_path / "cases.zip"
    if entries is None:
        entries = [("aligned/sample_001_pair_01.c", "int a() { return 1; }\n"),
                   ("aligned/sample_002_pair_01.c", "int a() { return 2; }\n"),
                   ("__MACOSX/._sample_001_pair_01.c", b"metadata")]
    with zipfile.ZipFile(path, "w") as handle:
        for name, content in entries:
            handle.writestr(name, content)
    return path


def test_import_preserves_sources_pairs_and_unknown_labels(tmp_path):
    source, output = archive(tmp_path), tmp_path / "evaluation"
    report = import_archive(source, output)
    cases = w.load_cases(output / "manifest.jsonl", output, "test", 1000)
    assert report["cases"] == 2 and report["pairs"] == 1
    assert cases[0]["pair_id"] == cases[1]["pair_id"]
    assert all(case["label"] is None and case["held_out"] for case in cases)
    assert cases[0]["sources"][0]["text"] == "int a() { return 1; }\n"
    assert cases[0]["sources"][0]["display_path"] == "review_target_001.c"
    assert not (output / "__MACOSX").exists()
    assert read_json(output / "import-report.json") == report
    with pytest.raises(FileExistsError):
        import_archive(source, output)
    with pytest.raises(ValueError, match="No cases in split calibration"):
        w.load_cases(output / "manifest.jsonl", output, "calibration", 1000)
    grades = tmp_path / "grades.json"
    write_new(grades, {"kind": "paired_review", "cases": cases})
    with pytest.raises(ValueError, match="reviewed clean/vulnerable labels"):
        w.evaluate(SimpleNamespace(input=grades, output=tmp_path / "metrics.json"))


@pytest.mark.parametrize("damage", ["path", "duplicate", "incomplete", "symlink", "encoding", "size"])
def test_invalid_zip_rejected_before_writing_sources(tmp_path, monkeypatch, damage):
    import freetoken.pruning.primevul as p

    first, second = "aligned/sample_001_pair_01.c", "aligned/sample_002_pair_01.c"
    data = b"int a;"
    if damage == "path":
        first = "../" + first
    elif damage == "duplicate":
        second = "other/sample_001_pair_01.c"
    elif damage == "symlink":
        first = zipfile.ZipInfo(first)
        first.external_attr = (stat.S_IFLNK | 0o777) << 16
    elif damage == "encoding":
        data = b"\xff"
    elif damage == "size":
        monkeypatch.setattr(p, "MAX_FILE_BYTES", 1)
    entries = [(first, data)] if damage == "incomplete" else [(first, data), (second, data)]
    source, output = archive(tmp_path, entries), tmp_path / "evaluation"
    with pytest.raises(ValueError):
        import_archive(source, output)
    assert not output.exists()


@pytest.mark.parametrize("damage", ["source", "split"])
def test_imported_evaluation_integrity_enforced_on_prepare(tmp_path, damage):
    output = tmp_path / "evaluation"
    import_archive(archive(tmp_path), output)
    manifest = output / "manifest.jsonl"
    if damage == "source":
        (output / "sources/sample_001_pair_01.c").write_text("changed")
    else:
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        rows[0]["split"] = "calibration"
        manifest.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="SHA-256|Held-out"):
        w.load_cases(manifest, output, "test", 1000)
