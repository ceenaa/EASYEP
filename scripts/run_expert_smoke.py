#!/usr/bin/env python3
"""Generate two real reasoning reviews on the live V4 expert-tracing server."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'FreeToken/python'))
from evaluate_full_model import align_dataset, json_review, now, save, source_prompt, stream_review
from freetoken.pruning.artifacts import checkpoint_identity
from freetoken.pruning.workflow import check_server, token_digest, tokenizer_tools


def main(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'streams').mkdir(exist_ok=True)
    selected = json.loads(Path(args.selection).read_text())['cases']
    actual = {c['id']: c for c in align_dataset(ROOT / 'primevul_aligned_100_samples 2.zip', ROOT / 'metadata_full.csv')}
    assert len(selected) == 2 and {c['truth'] for c in selected} == {'SAFE', 'VULNERABLE'}
    instruction = (ROOT / 'v03_extended.txt').read_text()
    tokenizer, manager = tokenizer_tools(args.model_dir)
    assert args.reasoning_effort in manager.effort_profile().supported
    checkpoint = checkpoint_identity(args.model_dir)
    model, metadata = check_server(args.server, checkpoint, {'statistics'})
    assert metadata['collection'] == 'live_prefill_decode' and metadata['mask_sha256'] is None
    prepared = {'kind': 'expert_monitoring_smoke', 'checkpoint': checkpoint,
                'instruction_sha256': hashlib.sha256(instruction.encode()).hexdigest(),
                'thinking_mode': 'thinking', 'reasoning_effort': args.reasoning_effort,
                'sampling': {'temperature': 0, 'top_p': 1, 'top_k': 1, 'max_tokens': args.max_tokens},
                'server': metadata, 'cases': []}
    for supplied in selected:
        case = actual[supplied['id']]
        assert supplied['source_sha256'] == case['source_sha256'] and supplied['truth'] == case['truth']
        raw = manager.render_prompt(SimpleNamespace(text=[{'role': 'user', 'content': source_prompt(case, instruction)}],
            tools=None, chat_template_kwargs={'thinking_mode': 'thinking', 'reasoning_effort': args.reasoning_effort}))
        assert raw.endswith('<think>')
        ids = tokenizer.encode(raw, add_special_tokens=True)
        numbered = '\n'.join(f'{i}: {line}' for i, line in enumerate(case['source'].splitlines(), 1))
        code_start = raw.index(numbered)
        code_end = code_start + len(numbered)
        encoded = tokenizer(raw, add_special_tokens=True, return_offsets_mapping=True)
        assert encoded['input_ids'] == ids
        source_positions = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if a < code_end and b > code_start]
        assert source_positions
        assert len(ids) + args.max_tokens < 16384
        prepared['cases'].append({**case, 'raw_prompt': raw, 'prompt_token_ids': ids,
                                 'prompt_tokens': len(ids), 'source_token_positions': source_positions,
                                 'source_character_span': [code_start, code_end], 'token_sha256': token_digest(ids)})
    assert not (output / 'prepared.json').exists(), 'Use a fresh smoke output directory'
    save(output / 'prepared.json', prepared)
    result = {'started_at': now(), 'complete': False, 'cases': []}
    save(output / 'reviews.json', result)
    for case in prepared['cases']:
        before = json.loads(Path(args.statistics).read_text())
        seen = set(before['requests'])
        record = {key: case[key] for key in ('id', 'filename', 'truth', 'source_sha256', 'prompt_tokens', 'token_sha256')}
        record.update(started_at=now(), status='error', server=metadata)
        events = output / 'streams' / (case['id'] + '.jsonl')
        print(json.dumps({'event': 'smoke_started', 'id': case['id'], 'at': now(),
                          'prompt_tokens': case['prompt_tokens'], 'reasoning_effort': args.reasoning_effort}), flush=True)
        try:
            response = stream_review(args.server, {'model': model, 'prompt': case['raw_prompt'],
                **prepared['sampling'], 'stream': True, 'stream_options': {'include_usage': True}}, events, 7200)
            record.update(response, events_file=str(events.relative_to(output)))
            assert response['stream_done'] and response['finish_reason'] == 'stop'
            assert response['usage']['prompt_tokens'] == case['prompt_tokens']
            record['structured_review'] = json_review(response['text'])
            record['prediction'] = {'vulnerable': 'VULNERABLE', 'not_vulnerable': 'SAFE'}[record['structured_review']['verdict']]
            stats = json.loads(Path(args.statistics).read_text())
            new = set(stats['requests']) - seen
            assert len(new) == 1, 'Expected exactly one real traced request'
            uid = new.pop()
            request = stats['requests'][uid]
            assert request['finished'] and request['finish_reason'] == 'stop'
            assert request['prefill_tokens'] == case['prompt_tokens']
            assert request['decode_tokens'] == response['usage']['completion_tokens'] - 1
            assert len(request['generated_token_ids']) == response['usage']['completion_tokens']
            _, current = check_server(args.server, checkpoint, {'statistics'})
            assert current == metadata
            record.update(status='complete', trace_uid=uid,
                          output_tps=response['usage']['completion_tokens'] / response['elapsed_seconds'])
            save(output / ('statistics_after_' + case['id'] + '.json'), stats)
        except Exception as error:
            record['error'] = f'{type(error).__name__}: {error}'
        record['finished_at'] = now()
        result['cases'].append(record)
        save(output / 'reviews.json', result)
        print(json.dumps({'event': 'smoke_finished', 'id': case['id'], 'status': record['status'],
                          'prediction': record.get('prediction'), 'output_tps': record.get('output_tps'),
                          'error': record.get('error')}), flush=True)
        if record['status'] != 'complete':
            raise RuntimeError(record['error'])
    result.update(complete=True, finished_at=now())
    save(output / 'reviews.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--statistics', required=True)
    parser.add_argument('--selection', default=str(ROOT / 'runs/easyep-smoke/input/selection.json'))
    parser.add_argument('--server', default='http://127.0.0.1:1919')
    parser.add_argument('--reasoning-effort', default='low')
    parser.add_argument('--max-tokens', type=int, default=8192)
    main(parser.parse_args())
