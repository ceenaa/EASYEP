# Run the prepared V4 EASYEP experiment

The V4 adapter has now run on an RTX 5090: all 74 pruning tests passed, and live expert traces were verified during input processing, reasoning and final answers for two code files. See [the smoke report](runs/easyep-smoke/REPORT.md) for results and review-quality limitations.

For a new rental, use the automatic Mac entry point: `./setup_vast_5090.sh NEW_SSH_HOST NEW_SSH_PORT`. It transfers the current local repositories, installs the recorded environment, downloads the pinned model, runs the reasoning smoke test and saves evidence locally. Requirements and options are in [REBUILD_5090.md](REBUILD_5090.md). The minimum allocated RAM check is 240 GB; the tested instance had approximately 513 GiB. Provision 300 GB or more disk with at least 240 GB free initially.

The remaining sections describe the manual calibration/pruning workflow. The current full evaluation remains paused, and the automatic rebuild does not perform pruning.

## 1. Transfer and install

Copy `dist/deepseek-v4-easyep-ready.tar.gz` to the instance using your SSH connection, extract it, and enter its `prunungdsk4` directory. The bundle includes the modified FreeToken source, tests, launcher and documentation; it excludes model weights, source-review datasets, virtual environments and Git history. Use a fresh environment so an older EASYEP/SGLang installation cannot replace FreeToken's dependencies.

```bash
tar -xzf deepseek-v4-easyep-ready.tar.gz
cd prunungdsk4
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e './FreeToken[accel]' pytest
uv pip freeze > installed-packages.txt
python scripts/inspect_target.py > target-before-load.json
PYTHONPATH=FreeToken/python python -m pytest --confcutdir FreeToken/tests/pruning FreeToken/tests/pruning -q
```

On the GPU, all pruning tests must pass; CUDA tests must run rather than skip. Stop before the large checkpoint load if those tests fail. The five CUDA checks cover two FP4 execution paths, hyper-connection orientation, actual V4 gate masking, and decode observation/output parity.

If the original model already exists on this instance, reuse its complete official checkpoint directory. Otherwise resolve and record a revision before downloading:

```bash
export MODEL_DIR="$PWD/models/DeepSeek-V4-Flash-0731"
python - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
repo = 'deepseek-ai/DeepSeek-V4-Flash-0731'
revision_file = Path('checkpoint-revision.json')
if revision_file.exists():
    record = json.loads(revision_file.read_text())
    assert record['repo'] == repo
else:
    record = {'repo': repo, 'revision': HfApi().model_info(repo).sha}
    revision_file.write_text(json.dumps(record, indent=2) + '\n')
snapshot_download(repo_id=repo, revision=record['revision'], local_dir=os.environ['MODEL_DIR'])
PY
python scripts/inspect_target.py --model-dir "$MODEL_DIR" > target.json
```

Review `target.json`: there should be 43 MoE layers, 256 experts, top-6 routing, hash layers 0-2, and no missing shards. The host inspector reports available data, not an assurance that the provider has allocated all host RAM to your instance. Preserve `checkpoint-revision.json`, `installed-packages.txt`, server logs, source manifest and all result artifacts.

## 2. Prepare source cases before inference

Set `MODEL_DIR` to the actual model directory in every terminal. Supply a source root and one JSONL manifest containing all splits. Each record has this shape (one JSON object per physical line):

```json
{"id":"project-a-001","repository":"project-a","revision":"source-commit","split":"calibration","language":"python","files":["project-a/src/component.py"],"label":"vulnerable","expected_findings":[{"id":"finding-1","category":"authorization","location":"project-a/src/component.py:40","explanation":"Reviewer-confirmed issue"}]}
```

For clean cases, use `"label":"clean","expected_findings":[]`. Use separate repositories for `calibration`, `validation`, and `test`. Related revisions belong in the same split. Preparation rejects exact source duplicates across splits in the manifest; a human still needs to check near duplicates and adequate coverage. Gold findings are retained for evaluation and excluded from prompts. Model-visible names are neutral, such as `review_target_001.py`; original paths and their display-name mapping stay in metadata. Files are read as UTF-8 and never executed. Checkpoint tokenizer/encoding code is the same code FreeToken normally loads.

The supplied PrimeVul ZIP has been imported locally as 100 held-out test cases. Follow [PRIMEVUL_EVALUATION.md](PRIMEVUL_EVALUATION.md) to transfer and prepare it. Ground-truth labels/repository metadata and separate calibration/validation data remain needed. Re-run `prepare` for any artifacts made before the filename fix; legacy prepared prompts are rejected.

```bash
mkdir -p results
python -m freetoken.pruning prepare --model-dir "$MODEL_DIR" --manifest cases.jsonl --source-root /path/to/source-snapshots --split calibration --output results/calibration.json
python -m freetoken.pruning prepare --model-dir "$MODEL_DIR" --manifest cases.jsonl --source-root /path/to/source-snapshots --split validation --output results/validation.json
python -m freetoken.pruning all-ones --model-dir "$MODEL_DIR" --output results/all-ones.json
```

The default prompt limit is 4,096 tokens, output budget 2,048 and server context 8,192. Oversized cases and unfinished reviews fail explicitly. If a review needs a larger budget, increase the relevant limits together; do not silently truncate. Begin with a few representative calibration and validation cases as a pilot, then use new output files for the larger experiment.

## 3. Baseline and all-ones parity

In terminal A:

```bash
python scripts/start_pruning_server.py baseline --model-dir "$MODEL_DIR"
```

Wait for `/health` to report `status: ok`. In terminal B, from the same directory and environment:

```bash
python -m freetoken.pruning reviews --input results/calibration.json --output results/calibration-baseline.json
python -m freetoken.pruning reviews --input results/validation.json --output results/validation-baseline.json
```

Stop the server with Ctrl-C in terminal A and wait for it to release GPU memory. Start the all-ones run:

```bash
python scripts/start_pruning_server.py mask --model-dir "$MODEL_DIR" --artifact results/all-ones.json
```

In terminal B:

```bash
python -m freetoken.pruning reviews --input results/validation.json --output results/validation-all-ones.json
python -m freetoken.pruning compare --baseline results/validation-baseline.json --candidate results/validation-all-ones.json --output results/parity.json
python - <<'PY'
import json
cases = json.load(open('results/parity.json'))['cases']
different = [case['id'] for case in cases if case['baseline'] != case['candidate']]
assert not different, f'All-ones output differs: {different}. Investigate before calibration.'
print('All-ones completion text matches on', len(cases), 'cases')
PY
```

This is a completion-level smoke check alongside the actual-gate unit test, not an exhaustive full-model logit-equivalence proof. Investigate differences before reducing the expert pool. Ordinary serving and all-ones mode leave the original FFN output arithmetic unchanged.

## 4. Collect EASYEP-style importance

Stop the all-ones server. Start a fresh statistics server in terminal A:

```bash
python scripts/start_pruning_server.py statistics --model-dir "$MODEL_DIR" --artifact results/stats.json
```

In terminal B:

```bash
python -m freetoken.pruning replay --model-dir "$MODEL_DIR" --input results/calibration-baseline.json --output results/replay.json
```

Use this server only for the replay client. It resubmits each rendered baseline prompt followed by the model's completed review text and requests one output token. Scoring covers only replay prefill tokens; the requested output token is not included. These are retokenized review texts, not a claim of recovering the original generated token IDs. Review API output retains the completion text returned by FreeToken.

Warm-up runs are outside the collector. Statistics mode requires eager execution, no prefix reuse, one request at a time and GPU offload. A batch commits only after all 43 layers are observed. Atomic snapshots record logical expert IDs, counts, gate sums, scores, per-request token hashes, and adapter source/package provenance. Restart failed collection with a new stats path and replay log: there is no automatic resume or request retry.

Stop the statistics server after successful replay. Generate candidate masks without loading a model:

```bash
python -m freetoken.pruning mask --stats results/stats.json --replay-log results/replay.json --keep 224 --output results/mask-224.json
python -m freetoken.pruning mask --stats results/stats.json --replay-log results/replay.json --keep 192 --output results/mask-192.json
```

Mask generation verifies the complete replay's token hashes/counts against the collected requests and per-layer activation counts. Extra traffic, missing chunks, or incomplete logs are rejected. Masks include per-layer expert rankings. Hash layers 0-2 and all shared experts remain intact; learned layers 3-42 retain the requested count. Selection excludes masked experts before top-k using negative infinity; original IDs and gate normalization remain unchanged.

The metric is named `easyep_v4_hc_weighted_v2`: sum over tokens of `norm(already gate-weighted expert output) * norm(post) * (1 - cosine(residual-only HC output, residual-plus-routed HC output))`, flattening V4's residual streams and excluding the shared expert. The `norm(post)` factor expands the expert norm into residual space without applying the gate twice. The cosine uses existing HC kernel outputs cast to FP32; this is not bitwise equivalence to the PR's all-FP32 reference. V1 statistics and masks tagged with the old metric are rejected: collect fresh statistics and regenerate masks. This is an experimental V4 extension, not a reproduction of EASYEP's R1 result. Config/index hashes bind artifacts to the loaded checkpoint structure; they do not hash every weight byte. Keep the pinned checkpoint immutable throughout.

## 5. Evaluate before selecting a mask

Start one candidate at a time in terminal A:

```bash
python scripts/start_pruning_server.py mask --model-dir "$MODEL_DIR" --artifact results/mask-224.json
```

In terminal B:

```bash
python -m freetoken.pruning reviews --input results/validation.json --output results/validation-224.json
python -m freetoken.pruning compare --baseline results/validation-baseline.json --candidate results/validation-224.json --output results/paired-224.json
```

Have a reviewer fill `judgment` for every case in a copy of `paired-224.json`: matched gold-finding IDs, false-positive counts, and notes on evidence, repairs and critical misses. Match by cause and location, not keywords. Then:

```bash
python -m freetoken.pruning evaluate --input results/paired-224-graded.json --output results/metrics-224.json
```

Repeat for other retention levels using the same prompts, sampling and server profile. Choose acceptable recall and false-positive loss before selecting a mask. Check rare and critical defect families separately; a small sample cannot establish that all cybersecurity capability is preserved. Select on validation, then prepare and evaluate the untouched test split once. The paired output records wall time per request, which includes HTTP overhead and is not a kernel-throughput benchmark.

Masking does **not** reduce the host expert bank size or checkpoint size, and whole-layer prefill transfers may remain unchanged. Physical weight pruning and a matching compact loader are a later implementation after quality validation. An inactive expert on these examples is not proven unnecessary for all security reviews.

## Cost and stopping

The launch profile uses one GPU, serial host loading, 512 cache slots, 256-token prefill chunks, an 8,192-token context, no CUDA graphs and no prefix reuse. These are initial experiment settings, not measured optimums for the rental. Use `--dry-run` to inspect the command without launching; adjust only with measured memory information. CPU/hybrid statistics collection is rejected.

The server listens on `127.0.0.1:1919`; run the client on the instance or through your SSH tunnel. Nothing rents, starts or stops a Vast instance automatically. At the quoted $0.80/hour, compute cost is elapsed hours times $0.80, plus any provider storage/network charges. **Ctrl-C stops FreeToken; stop the Vast instance separately when finished to stop its compute billing.**
