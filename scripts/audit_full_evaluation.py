#!/usr/bin/env python3
"""Independently check persisted evaluation records against inputs and raw SSE evidence."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(args):
    prepared = json.loads(Path(args.prepared).read_text())
    root = Path(args.run_dir)
    run = json.loads((root / 'run.json').read_text())
    settings = run['settings']
    assert settings['prepared_sha256'] == sha(args.prepared), 'Wrong prepared file for run'
    assert settings['instruction_sha256'] == prepared['instruction_sha256'] == sha(args.prompt), 'Wrong review prompt'
    assert prepared['archive_sha256'] == sha(args.archive), 'Dataset archive changed'
    assert prepared['metadata_sha256'] == sha(args.metadata), 'Ground-truth CSV changed'
    assert settings['reasoning_effort'] == prepared['reasoning_effort'] == 'max'
    assert settings['thinking_mode'] == prepared['thinking_mode'] == 'thinking'
    assert settings['output_schema'] == prepared['output_schema'] == 'json_audit'
    assert settings['temperature'] == 0 and settings['top_p'] == 1 and settings['top_k'] == 1
    if args.runner:
        assert settings['runner_sha256'] == sha(args.runner), 'Runner differs from recorded run'
    prompt = Path(args.prompt).read_text()
    assert prepared['instruction'] == prompt
    with Path(args.metadata).open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    truth = {int(row['sample']): row for row in rows}
    assert len(truth) == len(rows) == 100
    expected = {case['id']: case for case in prepared['cases']}
    assert len(expected) == len(prepared['cases']) == 100
    with zipfile.ZipFile(args.archive) as archive:
        for case in expected.values():
            row = truth[case['sample']]
            data = archive.read(case['archive_member'])
            assert hashlib.sha256(data).hexdigest() == case['source_sha256'] == row['source_sha256']
            assert case['truth'] == row['truth'] and case['pair'] == int(row['pair'])
            numbered = '\n'.join(f'{i}: {line}' for i, line in enumerate(data.decode().splitlines(), 1))
            assert prompt + '\n\nFILE ' + case['display_path'] + '\n' + numbered in case['raw_prompt']
            assert 'Reasoning Effort: Beyond maximum' in case['raw_prompt'] and case['raw_prompt'].endswith('<think>')
            assert case['filename'] not in case['raw_prompt'] and row['filename'] not in case['raw_prompt']
            assert row['cve'] not in case['raw_prompt']

    records = {p.stem: json.loads(p.read_text()) for p in (root / 'cases').glob('*.json')}
    assert records.keys() <= expected.keys(), 'Unexpected case record'
    counts, failures, errors, output_tokens, seconds = Counter(), [], [], 0, 0.0
    for case_id, record in records.items():
        case = expected[case_id]
        for key in ('id', 'sample', 'pair', 'truth', 'source_sha256', 'token_sha256', 'prompt_tokens'):
            assert record[key] == case[key], (case_id, 'input mismatch', key)
        assert record['server']['mode'] == 'baseline' and record['server']['mask_sha256'] is None
        assert record['server']['checkpoint'] == prepared['checkpoint'] == run['checkpoint']
        assert record['max_tokens'] == min(settings['max_tokens'], settings['max_seq_len'] - case['prompt_tokens'] - 1)
        if record['status'] != 'complete':
            errors.append({'file': case['filename'], 'error': record.get('error')})
            continue
        events_path = (root / record['events_file']).resolve()
        assert events_path.is_relative_to(root.resolve())
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        times = [event['elapsed_seconds'] for event in events]
        assert times and times == sorted(times) and times[0] >= 0
        pieces, usage, finish, text_times = [], None, None, []
        for item in events:
            event = item['event']
            assert not event.get('error'), (case_id, 'stream error')
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                if choice.get('text'):
                    pieces.append(choice['text'])
                    text_times.append(item['elapsed_seconds'])
                if choice.get('finish_reason') is not None:
                    finish = choice['finish_reason']
        text = ''.join(pieces)
        assert text == record['text'] and record['stream_done']
        assert finish == record['finish_reason'] == 'stop'
        assert usage == record['usage'] and usage['prompt_tokens'] == case['prompt_tokens']
        assert usage['completion_tokens'] > 0
        assert '</think>' in text
        final = text.rsplit('</think>', 1)[1].strip()
        final = re.sub(r'(?:<[|｜][^<>]+[|｜]>\s*)+$', '', final).strip()
        if final.startswith('```'):
            lines = final.splitlines()
            assert lines[0] in ('```json', '```') and lines[-1] == '```'
            final = '\n'.join(lines[1:-1])
        structured = json.loads(final)
        assert structured == record['structured_review']
        prediction = {'vulnerable': 'VULNERABLE', 'not_vulnerable': 'SAFE'}[structured['verdict']]
        assert prediction == record['prediction']
        assert text_times[0] == record['ttft_seconds'] and text_times[-1] == record['last_text_seconds']
        assert record['elapsed_seconds'] >= times[-1]
        assert math.isclose(record['output_tps_e2e'], usage['completion_tokens'] / record['elapsed_seconds'], rel_tol=1e-9)
        span = text_times[-1] - text_times[0]
        if span > 0:
            assert math.isclose(record['decode_tps_estimate'], (usage['completion_tokens'] - 1) / span, rel_tol=1e-9)
        key = ('T' if prediction == case['truth'] else 'F') + ('P' if prediction == 'VULNERABLE' else 'N')
        counts[key] += 1
        if key in ('FP', 'FN'):
            failures.append({'file': case['filename'], 'outcome': key, 'truth': case['truth'], 'prediction': prediction})
        output_tokens += usage['completion_tokens']
        seconds += record['elapsed_seconds']
    metrics = json.loads((root / 'metrics.json').read_text()) if (root / 'metrics.json').exists() else None
    if metrics:
        assert all(metrics[key] == counts[key] for key in ('TP', 'TN', 'FP', 'FN'))
        assert metrics['errors'] == len(errors) and metrics['pending'] == len(expected) - len(records)
        if seconds:
            assert math.isclose(metrics['output_tps_e2e_weighted'], output_tokens / seconds, rel_tol=1e-9)
    result = {'verified_complete': len(records) == 100 and not errors,
              'verified_finished_cases': sum(counts.values()), 'pending': len(expected) - len(records),
              'counts': {key: counts[key] for key in ('TP', 'TN', 'FP', 'FN')},
              'errors': errors, 'misclassified_files': failures,
              'output_tokens': output_tokens, 'request_seconds_completed': seconds,
              'weighted_output_tps_e2e': output_tokens / seconds if seconds else None,
              'prompt_sha256': sha(args.prompt), 'prepared_sha256': sha(args.prepared),
              'checks': 'All 100 source/CSV/prompt alignments; run hashes; per-record input identity; baseline mode; reconstructed SSE text, usage, final JSON verdict and timing; independently counted confusion matrix.'}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('verified_complete', 'verified_finished_cases', 'pending', 'counts', 'weighted_output_tps_e2e')}, indent=2))
    if not args.allow_partial and not result['verified_complete']:
        raise SystemExit('Evaluation is not complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'prepared', 'prompt', 'archive', 'metadata', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--runner')
    parser.add_argument('--allow-partial', action='store_true')
    audit(parser.parse_args())
