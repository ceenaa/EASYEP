# Supplied PrimeVul evaluation corpus

Imported `primevul_aligned_100_samples 2.zip` on 2026-09-10 into `data/primevul-evaluation/`. The original ZIP is unchanged. Source files were read as data and never compiled or executed.

| Property | Verified value |
|---|---|
| Source cases | 100 |
| Pairs from filenames | 50, two files per pair |
| Languages | 72 C, 28 C++ |
| Source bytes | 362,925 |
| Largest case | 22,967 bytes |
| Identical complete source files | None |
| Split | All `test`, marked `held_out: true` |
| Ground-truth labels / original repositories | Not supplied in the ZIP |
| ZIP SHA-256 | `fb7a0755d47a1433bf1582d5cc10cf34e68d23af310a20fcfa3f33ba490e54ec` |

`manifest.jsonl` records case IDs, pair IDs, original archive members, source hashes and the test split. `import-report.json` records the inventory and provenance. macOS resource forks are omitted. Preparation verifies the source hashes and rejects reassignment of these cases to calibration or validation.

The filenames establish pairs but do not establish which member is vulnerable. Labels remain `null`, and empty `expected_findings` lists mean **unannotated**, not verified clean. Do not infer labels from odd/even sample numbers. Supply the original label/repository/revision mapping and reviewer-confirmed findings before computing quality metrics. A fixed version is not automatically free of unrelated defects. The existing evaluator rejects missing labels and unfilled judgments.

All 100 cases are reserved for final evaluation. Use separate source cases for expert activation collection and retention selection. Original repositories are unknown, so repository and near-duplicate separation from future calibration data still needs checking. The manifest's `unknown:primevul-aligned` repository value is an explicit placeholder, not an upstream repository identity. Checks across splits apply to cases in the same input manifest; separate manifests do not establish isolation by themselves.

## Transfer and prepare

The code deployment bundle includes the importer and this guide; it excludes source datasets. Transfer the local `data/primevul-evaluation/` directory separately, or copy the original ZIP to the instance and reproduce the import from the unpacked project root:

```bash
python -m freetoken.pruning import-primevul --archive 'primevul_aligned_100_samples 2.zip' --output-dir data/primevul-evaluation
```

The importer refuses to overwrite an existing directory. Fill the manifest's labels, original repositories/revisions, and `expected_findings` using verified ground truth. Preserve the source hashes and `held_out` flag. The model receives only numbered source lines under names such as `review_target_001.c`; case IDs, pair IDs, original paths and labels are kept in local metadata. Original source text is preserved. The baseline and masked model review each function independently; the paired version is not added to its prompt.

Once the checkpoint/tokenizer is installed, prepare the test prompts:

```bash
mkdir -p results
python -m freetoken.pruning prepare --model-dir "$MODEL_DIR" --manifest data/primevul-evaluation/manifest.jsonl --source-root data/primevul-evaluation --split test --max-prompt-tokens 6000 --output results/primevul-test.json
```

Native token counts have not been measured locally. The 6,000-token input limit leaves room for the default 2,048-token completion in the 8,192-token context. Oversized functions fail without truncation; increase the input/context limits consistently if needed. Generating reviews can proceed with unknown labels, but metric calculation cannot.

## Evaluate the selected mask

First pass the CUDA tests and baseline/all-ones pilot in [RUN_ON_VAST.md](RUN_ON_VAST.md), using separate calibration/validation cases. Select the retained expert count on validation before evaluating these final test cases.

With the unmasked baseline server running:

```bash
python -m freetoken.pruning reviews --input results/primevul-test.json --output results/primevul-baseline.json
```

Restart the server with the mask selected on validation, then run:

```bash
python -m freetoken.pruning reviews --input results/primevul-test.json --output results/primevul-masked.json
python -m freetoken.pruning compare --baseline results/primevul-baseline.json --candidate results/primevul-masked.json --output results/primevul-paired.json
```

The comparison retains original-to-display path mappings and corpus pair IDs for the reviewer. Here `baseline`/`candidate` refers to two model variants reviewing the same function, not the two source versions in a PrimeVul pair. Reviewers fill matched finding IDs and false-positive counts in a copy, `results/primevul-paired-graded.json`, using verified labels and findings:

```bash
python -m freetoken.pruning evaluate --input results/primevul-paired-graded.json --output results/primevul-metrics.json
```

This produces human-graded finding precision/recall and clean-case false positives. It does not automatically produce the official PrimeVul binary/paired benchmark metrics. Neither model reviews nor quality measurements have been run on these files yet.
