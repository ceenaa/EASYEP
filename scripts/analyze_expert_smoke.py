#!/usr/bin/env python3
"""Verify real prompt/reasoning/answer traces and summarize expert activity."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'FreeToken/python'))
from evaluate_full_model import align_dataset, json_review, save, source_prompt
from freetoken.pruning import METRIC
from freetoken.pruning.workflow import token_digest


def main(args):
    root = Path(args.run_dir)
    prepared = json.loads((root / 'prepared.json').read_text())
    reviews = json.loads((root / 'reviews.json').read_text())
    stats = json.loads((root / 'statistics.json').read_text())
    assert reviews['complete'] and len(reviews['cases']) == 2
    assert stats['collection'] == 'live_prefill_decode' and stats['metric'] == METRIC
    assert prepared['server']['collection'] == 'live_prefill_decode'
    assert prepared['server']['mode'] == 'statistics' and prepared['server']['mask_sha256'] is None
    assert prepared['checkpoint'] == stats['checkpoint']
    geometry = stats['checkpoint']['geometry']
    layers, experts, topk = (geometry[k] for k in ('n_layers', 'n_routed_experts', 'n_activated_experts'))
    assert (layers, experts, topk) == (43, 256, 6)
    # Audit saved IDs/offsets only; do not import the CUDA inference stack.
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    closing = tokenizer.encode('</think>', add_special_tokens=False)
    expected = {c['id']: c for c in align_dataset(ROOT / 'primevul_aligned_100_samples 2.zip', ROOT / 'metadata_full.csv')}
    inputs = {c['id']: c for c in prepared['cases']}
    instruction = (ROOT / 'v03_extended.txt').read_text()
    assert hashlib.sha256(instruction.encode()).hexdigest() == prepared['instruction_sha256']
    for name, digest in stats['provenance']['source_sha256'].items():
        assert hashlib.sha256((ROOT / 'FreeToken/python/freetoken' / name).read_bytes()).hexdigest() == digest
    phases = ('reading', 'code_only', 'reasoning', 'answer', 'continuation_replay')
    state = {}
    requests = {}
    for record in reviews['cases']:
        case = inputs[record['id']]
        assert record['status'] == 'complete' and record['finish_reason'] == 'stop' and record['stream_done']
        assert record['source_sha256'] == case['source_sha256'] == expected[case['id']]['source_sha256']
        assert record['truth'] == expected[case['id']]['truth']
        assert token_digest(case['prompt_token_ids']) == case['token_sha256'] == record['token_sha256']
        assert tokenizer.encode(case['raw_prompt'], add_special_tokens=True) == case['prompt_token_ids']
        assert source_prompt(expected[case['id']], instruction) in case['raw_prompt']
        assert case['filename'] not in case['raw_prompt'] and case['ground_truth']['cve'] not in case['raw_prompt']
        encoded = tokenizer(case['raw_prompt'], add_special_tokens=True, return_offsets_mapping=True)
        begin, end = case['source_character_span']
        numbered = '\n'.join(f'{i}: {line}' for i, line in enumerate(expected[case['id']]['source'].splitlines(), 1))
        assert case['raw_prompt'][begin:end] == numbered
        assert case['source_token_positions'] == [i for i, (a, b) in enumerate(encoded['offset_mapping']) if a < end and b > begin]
        assert case['raw_prompt'].endswith('<think>') and prepared['thinking_mode'] == 'thinking'
        segments = record.get('trace_segments') or [{'trace_uid': record['trace_uid'],
            'prompt_token_ids': case['prompt_token_ids'], 'events_file': record['events_file'], 'response': record}]
        generated, streamed = [], []
        item = {'case': case, 'record': record, 'tokens': [], 'segments': len(segments),
                'phases': {phase: {'tokens': 0, 'counts': np.zeros((layers, experts), dtype=np.int64),
                    'gate_sums': np.zeros((layers, experts)), 'scores': np.zeros((layers, experts))}
                    for phase in phases}}
        for index, segment in enumerate(segments):
            uid = segment['trace_uid']
            request = stats['requests'][uid]
            response = segment['response']
            prefix = case['prompt_token_ids'] + generated
            assert segment['prompt_token_ids'] == prefix
            if index:
                raw = case['raw_prompt'] + tokenizer.decode(generated, skip_special_tokens=False)
                assert tokenizer.encode(raw, add_special_tokens=True) == prefix
            expected_finish = 'stop' if index == len(segments) - 1 else 'length'
            assert request['finished'] and request['finish_reason'] == response['finish_reason'] == expected_finish
            assert response['stream_done']
            assert request['prefill_tokens'] == response['usage']['prompt_tokens'] == len(prefix)
            assert len(request['generated_token_ids']) == response['usage']['completion_tokens']
            assert request['decode_tokens'] == len(request['generated_token_ids']) - 1
            events = [json.loads(line)['event'] for line in (root / segment['events_file']).read_text().splitlines()]
            assert ''.join(c.get('text', '') for event in events for c in event.get('choices', [])) == response['text']
            assert [event['usage'] for event in events if event.get('usage')][-1] == response['usage']
            assert any(c.get('finish_reason') == expected_finish for event in events for c in event.get('choices', []))
            assert uid not in requests
            requests[uid] = {'item': item, 'tokens': [], 'expected_tokens': prefix + request['generated_token_ids'][:-1],
                             'prefill_tokens': len(prefix), 'replay_prefix': len(prefix) - 1 if index else 0}
            generated.extend(request['generated_token_ids'])
            streamed.append(response['text'])
        assert ''.join(streamed) == record['text']
        assert len(generated) == record['usage']['completion_tokens']
        assert case['prompt_tokens'] == record['usage']['prompt_tokens']
        matches = [i for i in range(len(generated)-len(closing)+1) if generated[i:i+len(closing)] == closing]
        assert len(matches) == 1 and matches[0] > 0, 'Need real reasoning and one closing marker'
        boundary = matches[0] + len(closing) - 1
        assert json_review(tokenizer.decode(generated, skip_special_tokens=False)) == record['structured_review']
        item.update(generated=generated, boundary=boundary)
        state[case['id']] = item
    assert set(requests) == set(stats['requests']), 'Unexpected inference requests in the trace'
    with (root / stats['trace_file']).open() as handle:
        for line in handle:
            event = json.loads(line)
            request = requests[event['uid']]
            item = request['item']
            case = item['case']
            count = len(event['token_ids'])
            start = len(request['tokens'])
            assert event['start_position'] == start
            request['tokens'].extend(event['token_ids'])
            ids = np.asarray(event['expert_ids'], dtype=np.int64)
            gates = np.asarray(event['routing_weights'], dtype=np.float32)
            norms = np.asarray(event['residual_weighted_output_norms'], dtype=np.float32)
            sensitivity = np.asarray(event['sensitivity'], dtype=np.float32)
            assert ids.shape == gates.shape == norms.shape == (layers, count, topk)
            assert sensitivity.shape == (layers, count)
            assert ((ids >= 0) & (ids < experts)).all()
            assert all(np.isfinite(a).all() and (a >= 0).all() for a in (gates, norms, sensitivity))
            assert (sensitivity <= 2).all()
            scores = norms * sensitivity[:, :, None]
            positions = np.arange(start, start + count)
            prefill = positions < request['prefill_tokens']
            assert event['phase'] in ('prefill', 'decode')
            assert prefill.all() if event['phase'] == 'prefill' else not prefill.any()
            replay = positions < request['replay_prefix']
            novel_positions = positions[~replay]
            assert novel_positions.tolist() == list(range(len(item['tokens']), len(item['tokens']) + len(novel_positions)))
            item['tokens'].extend(np.asarray(event['token_ids'])[~replay].tolist())
            reading = positions < case['prompt_tokens']
            masks = {'reading': reading & ~replay,
                     'code_only': np.isin(positions, case['source_token_positions']) & ~replay,
                     'reasoning': (~reading) & ~replay & (positions - case['prompt_tokens'] < item['boundary']),
                     'answer': (~reading) & ~replay & (positions - case['prompt_tokens'] >= item['boundary']),
                     'continuation_replay': replay}
            for phase, mask in masks.items():
                summary = item['phases'][phase]
                summary['tokens'] += int(mask.sum())
                for layer in range(layers):
                    selected = ids[layer, mask].reshape(-1)
                    np.add.at(summary['counts'][layer], selected, 1)
                    np.add.at(summary['gate_sums'][layer], selected, gates[layer, mask].reshape(-1))
                    np.add.at(summary['scores'][layer], selected, scores[layer, mask].reshape(-1))
    total_counts = np.zeros((layers, experts), dtype=np.int64)
    total_gates, total_scores = np.zeros((layers, experts)), np.zeros((layers, experts))
    summaries = []
    with (root / 'expert_activity.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['file', 'phase', 'layer', 'expert_id', 'activations', 'routing_weight_sum', 'importance'])
        for item in state.values():
            case, record = item['case'], item['record']
            assert item['tokens'] == case['prompt_token_ids'] + item['generated'][:-1]
            phases_out = {}
            for phase, values in item['phases'].items():
                assert values['tokens'] > 0 or phase == 'continuation_replay', (case['id'], phase, 'missing coverage')
                assert (values['counts'].sum(1) == values['tokens'] * topk).all()
                assert (values['scores'] >= 0).all()
                assert values['scores'].sum() > 0 if values['tokens'] else values['scores'].sum() == 0
                active = (values['counts'] > 0).sum(1)
                ranks = np.argsort(-values['scores'][3], kind='stable')[:5].tolist()
                phases_out[phase] = {'tokens': values['tokens'], 'layers_observed': layers if values['tokens'] else 0,
                    'expert_selections': int(values['counts'].sum()),
                    'active_experts_per_layer_min': int(active.min()),
                    'active_experts_per_layer_max': int(active.max()),
                    'layer_3_top_experts_by_importance': ranks}
                for layer in range(layers):
                    for expert in range(experts):
                        writer.writerow([case['filename'], phase, layer, expert,
                            int(values['counts'][layer, expert]), float(values['gate_sums'][layer, expert]),
                            float(values['scores'][layer, expert])])
                if phase != 'code_only':
                    total_counts += values['counts']
                    total_gates += values['gate_sums']
                    total_scores += values['scores']
            summaries.append({'id': case['id'], 'filename': case['filename'], 'truth': record['truth'],
                              'prediction': record['prediction'], 'phases': phases_out,
                              'request_segments': item['segments'],
                              'ground_truth_cwe': case['ground_truth']['cwe'],
                              'output_tokens': record['usage']['completion_tokens'],
                              'seconds': record['elapsed_seconds'], 'output_tps_with_tracing': record['output_tps'],
                              'review': record['structured_review']})
    np.testing.assert_array_equal(total_counts, stats['activation_counts'])
    np.testing.assert_allclose(total_gates, stats['gate_sums'], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(total_scores, stats['scores'], rtol=1e-4, atol=1e-3)
    for uid, request in requests.items():
        assert request['tokens'] == request['expected_tokens']
        assert token_digest(request['tokens']) == stats['requests'][uid]['token_sha256']
    assert stats['total_tokens'] == sum(len(request['tokens']) for request in requests.values())
    report = {'verified': True, 'scope': 'two-file monitoring smoke test; no pruning or expert selection',
              'metric': METRIC, 'thinking_mode': prepared['thinking_mode'], 'reasoning_effort': prepared['reasoning_effort'],
              'geometry': geometry, 'total_forwarded_tokens': stats['total_tokens'],
              'unique_review_forwarded_tokens': sum(len(item['tokens']) for item in state.values()),
              'continuation_replay_tokens': sum(item['phases']['continuation_replay']['tokens'] for item in state.values()),
              'total_expert_selections': int(total_counts.sum()), 'cases': summaries,
              'phase_definition': 'Reading covers original prompt forward passes. Code-only is an overlapping source-token subset. Generated-token forward passes before the closing thinking marker are reasoning; the pass consuming the closing marker starts the answer. The first generated token is predicted by the final prompt pass, and the final sampled token is not fed back. For an output-limit continuation, the duplicated prefix is counted separately as continuation replay; only the last prefix-token pass and new decode passes extend the unique review trace.'}
    save(root / 'verification.json', report)
    lines = ['# V4 EASYEP-style live expert monitoring smoke test', '',
             '**Verified:** real expert activations and output-aware scores were captured during input processing, reasoning, and final answers for both selected files.', '',
             f'Engine: FreeToken; full unmasked DeepSeek-V4-Flash-0731; native reasoning `{prepared["reasoning_effort"]}`; metric `{METRIC}`.', '',
             '| File | CSV truth | Model verdict | Reading tokens | Code-only tokens | Reasoning passes | Answer passes | Layers per phase |',
             '|---|---|---|---:|---:|---:|---:|---|']
    for case in summaries:
        p = case['phases']
        lines.append(f'| {case["filename"]} | {case["truth"]} | {case["prediction"]} | {p["reading"]["tokens"]} | {p["code_only"]["tokens"]} | {p["reasoning"]["tokens"]} | {p["answer"]["tokens"]} | 43/43 |')
    lines += ['', report['phase_definition'], '',
              'Each observed token selected six logical routed experts in each of 43 layers. Shared experts remain enabled and are excluded from routed-expert rankings. Label metadata was withheld from prompts. Source hashes match the supplied ZIP and CSV.', '',
              '## Model answers', '']
    for case in summaries:
        lines += [f'### {case["filename"]}', '', 'Model summary: ' + case['review']['summary'], '',
                  f'Output: {case["output_tokens"]} tokens in {case["seconds"]:.2f} seconds; {case["output_tps_with_tracing"]:.2f} tokens/s with live tracing enabled.', '']
        if case['truth'] == 'VULNERABLE':
            lines += [f'CSV weakness category: `{case["ground_truth_cwe"]}`. Model categories: `{", ".join(case["review"]["cwe"])}`. '
                      'A matching binary verdict does not establish that the reported mechanism matches the ground-truth weakness.', '']
        if case['request_segments'] > 1:
            lines += [f'This review used {case["request_segments"]} requests: the first stopped at its output limit and the next continued the exact saved token sequence. '
                      f'{case["phases"]["continuation_replay"]["tokens"]:,} duplicated prefix passes are recorded separately and excluded from the reading/reasoning/answer counts above. '
                      'The original truncated response is preserved in `reviews_before_continuation.json`; provider usage and streams remain available per segment.', '']
    lines += ['## Evidence', '', '- [Verification and per-phase summaries](verification.json).',
              '- [All per-file, per-phase, per-layer, per-expert counts and importance scores](expert_activity.csv).',
              '- [Token-level expert traces](statistics.trace.jsonl).',
              '- [Full responses and request timings](reviews.json).',
              '- [Exact prompts and source token boundaries](prepared.json).',
              '- [Aggregate collector statistics and source provenance](statistics.json).', '',
              'The analyzer reconstructed counts and importance from raw routes, checked all token IDs against the prompts and generated sequence, located the native reasoning boundary, matched final JSON to token decoding and saved stream events, and compared the reconstruction with the collector totals.', '',
              'This validates the experimental V4 monitoring integration. It does not validate pruning quality or reproduce the original EASYEP model experiments. Instrumentation adds overhead; these timings are not an optimized serving benchmark.', '']
    (root / 'SMOKE_TEST.md').write_text('\n'.join(lines))
    print(json.dumps({'verified': True, 'cases': [{k: c[k] for k in ('filename', 'truth', 'prediction', 'phases', 'output_tps_with_tracing')} for c in summaries]}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--model-dir', required=True)
    main(parser.parse_args())
