#!/usr/bin/env python3
"""Continue one length-limited smoke review, retaining every request and trace."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'FreeToken/python'))
from evaluate_full_model import json_review, now, save, stream_review
from freetoken.pruning.workflow import check_server, token_digest, tokenizer_tools


def main(args):
    root = Path(args.run_dir)
    prepared = json.loads((root / 'prepared.json').read_text())
    reviews = json.loads((root / 'reviews.json').read_text())
    stats = json.loads((root / 'statistics.json').read_text())
    failed = [c for c in reviews['cases'] if c['status'] == 'error']
    assert not reviews['complete'] and len(failed) == 1
    previous = failed[0]
    assert previous['finish_reason'] == 'length' and previous['stream_done']
    case = next(c for c in prepared['cases'] if c['id'] == previous['id'])
    known = {c['trace_uid'] for c in reviews['cases'] if c['status'] == 'complete'}
    failed_uids = set(stats['requests']) - known
    assert len(failed_uids) == 1
    prior_uid = failed_uids.pop()
    prior = stats['requests'][prior_uid]
    generated = prior['generated_token_ids']
    assert prior['finished'] and prior['finish_reason'] == 'length'
    assert prior['prefill_tokens'] == case['prompt_tokens']
    assert len(generated) == previous['usage']['completion_tokens']
    assert prior['decode_tokens'] == len(generated) - 1
    assert prior['token_sha256'] == token_digest(case['prompt_token_ids'] + generated[:-1])
    tokenizer, _ = tokenizer_tools(args.model_dir)
    raw = case['raw_prompt'] + tokenizer.decode(generated, skip_special_tokens=False)
    ids = tokenizer.encode(raw, add_special_tokens=True)
    assert ids == case['prompt_token_ids'] + generated, 'Continuation must preserve exact token IDs'
    assert '</think>' not in tokenizer.decode(generated, skip_special_tokens=False)
    assert len(ids) + args.max_tokens < 16384
    model, metadata = check_server(args.server, prepared['checkpoint'], {'statistics'})
    assert metadata == prepared['server']
    path = root / 'continuation.json'
    assert not path.exists(), 'Continuation already exists; inspect its saved status'
    save(root / 'reviews_before_continuation.json', reviews)
    save(root / 'statistics_before_continuation.json', stats)
    document = {'case_id': case['id'], 'started_at': now(), 'status': 'running',
                'prior_trace_uid': prior_uid, 'raw_prompt': raw, 'prompt_token_ids': ids,
                'token_sha256': token_digest(ids), 'max_tokens': args.max_tokens,
                'note': 'Exact token continuation after the original output limit; duplicated prefix is replay overhead.'}
    save(path, document)
    print(json.dumps({'event': 'continuation_started', 'id': case['id'], 'prompt_tokens': len(ids),
                      'max_tokens': args.max_tokens, 'at': now()}), flush=True)
    try:
        events = root / 'streams' / (case['id'] + '-continuation.jsonl')
        response = stream_review(args.server, {'model': model, 'prompt': raw,
            'temperature': 0, 'top_p': 1, 'top_k': 1, 'max_tokens': args.max_tokens,
            'stream': True, 'stream_options': {'include_usage': True}}, events, 7200)
        document.update(response=response, events_file=str(events.relative_to(root)))
        save(path, document)
        assert response['stream_done'] and response['finish_reason'] == 'stop'
        assert response['usage']['prompt_tokens'] == len(ids)
        after = json.loads((root / 'statistics.json').read_text())
        new = set(after['requests']) - set(stats['requests'])
        assert len(new) == 1
        uid = new.pop()
        request = after['requests'][uid]
        assert request['finished'] and request['finish_reason'] == 'stop'
        assert request['prefill_tokens'] == len(ids)
        assert len(request['generated_token_ids']) == response['usage']['completion_tokens']
        assert request['decode_tokens'] == response['usage']['completion_tokens'] - 1
        _, current = check_server(args.server, prepared['checkpoint'], {'statistics'})
        assert current == metadata
        combined = previous['text'] + response['text']
        review = json_review(combined)
        assert review == json_review(tokenizer.decode(generated + request['generated_token_ids'], skip_special_tokens=False))
        record = dict(previous)
        record.pop('error', None)
        total_output = len(generated) + len(request['generated_token_ids'])
        seconds = previous['elapsed_seconds'] + response['elapsed_seconds']
        record.update(status='complete', finish_reason='stop', text=combined,
            structured_review=review, prediction={'vulnerable': 'VULNERABLE', 'not_vulnerable': 'SAFE'}[review['verdict']],
            usage={'prompt_tokens': case['prompt_tokens'], 'completion_tokens': total_output,
                   'total_tokens': case['prompt_tokens'] + total_output},
            usage_scope='Logical combined review; actual provider usage is retained per segment.',
            elapsed_seconds=seconds, output_tps=total_output / seconds, finished_at=now(),
            trace_segments=[{'trace_uid': prior_uid, 'prompt_token_ids': case['prompt_token_ids'],
                             'events_file': previous['events_file'], 'response': previous},
                            {'trace_uid': uid, 'prompt_token_ids': ids,
                             'events_file': document['events_file'], 'response': response}])
        reviews['cases'] = [record if c['id'] == case['id'] else c for c in reviews['cases']]
        reviews.update(complete=all(c['status'] == 'complete' for c in reviews['cases']), finished_at=now())
        save(root / 'reviews.json', reviews)
        save(root / ('statistics_after_' + case['id'] + '.json'), after)
        document.update(status='complete', trace_uid=uid, finished_at=now())
        save(path, document)
        print(json.dumps({'event': 'continuation_complete', 'id': case['id'], 'prediction': record['prediction'],
                          'combined_output_tokens': total_output, 'seconds': seconds}), flush=True)
    except Exception as error:
        document.update(status='error', error=f'{type(error).__name__}: {error}', finished_at=now())
        save(path, document)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--server', default='http://127.0.0.1:1919')
    parser.add_argument('--max-tokens', type=int, default=7000)
    main(parser.parse_args())
