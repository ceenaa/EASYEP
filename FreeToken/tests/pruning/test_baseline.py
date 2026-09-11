import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('full_model_evaluation', ROOT / 'scripts/evaluate_full_model.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_only_terminal_verdict_is_scored():
    assert evaluation.verdict('Maybe VERDICT: SAFE\nReview: defect found\nVERDICT: VULNERABLE') == 'VULNERABLE'
    assert evaluation.verdict('VERDICT: SAFE\nBut I need more analysis.') is None
    assert evaluation.verdict('VERDICT: SAFE<|eos|>') == 'SAFE'
    assert evaluation.verdict('VERDICT: SAFE<｜end▁of▁sentence｜>') == 'SAFE'


def test_prompt_omits_label_metadata():
    case = {'source': 'int f(void) { return 0; }', 'display_path': 'review_target_001.c',
            'filename': 'CWE-vulnerable.c', 'truth': 'VULNERABLE', 'ground_truth': {'cve': 'CVE-secret'}}
    prompt = evaluation.source_prompt(case)
    assert 'CWE-vulnerable' not in prompt and 'CVE-secret' not in prompt
    assert 'FILE review_target_001.c\n1: int f' in prompt


def audit_result(verdict='not_vulnerable'):
    return {'audit': {key: 'Visible code facts.' for key in (
        'bounds_and_types', 'lifetime_and_state', 'security_contracts', 'counterevidence')},
        'summary': 'Assessment.', 'evidence': [{'lines': '1-2', 'explanation': 'Observed guard.'}],
        'cwe': [], 'recommendation': '', 'confidence': 0.7, 'verdict': verdict}


def test_custom_prompt_is_preserved_without_old_verdict_instruction():
    instruction = (ROOT / 'v03_extended.txt').read_text()
    case = {'source': 'int f(void) { return 0; }', 'display_path': 'review_target_001.c'}
    result = evaluation.source_prompt(case, instruction)
    assert result.startswith(instruction + '\n\nFILE review_target_001.c\n')
    assert 'VERDICT: SAFE' not in result


def test_json_audit_scores_final_answer_not_tentative_reasoning():
    result = audit_result()
    text = 'Considering ' + json.dumps(audit_result('vulnerable')) + '</think>\n' + json.dumps(result)
    assert evaluation.json_review(text) == result
    assert evaluation.json_review('Thinking.</think>\n```json\n' + json.dumps(result) + '\n```<｜end▁of▁sentence｜>') == result


@pytest.mark.parametrize('damage', ['thinking_only', 'duplicate', 'confidence', 'verdict', 'audit', 'trailing'])
def test_invalid_json_audit_is_not_silently_scored(damage):
    result = audit_result()
    if damage == 'confidence':
        result['confidence'] = True
    elif damage == 'verdict':
        result['verdict'] = 'maybe'
    elif damage == 'audit':
        del result['audit']['counterevidence']
    text = json.dumps(result)
    if damage == 'duplicate':
        text = text[:-1] + ', "verdict": "vulnerable"}'
    if damage == 'trailing':
        text += '\nAdditional prose.'
    if damage != 'thinking_only':
        text = 'Analysis.</think>' + text
    with pytest.raises(ValueError):
        evaluation.json_review(text)


def test_metrics_separate_errors_and_missing_cases_from_false_negatives(tmp_path):
    prepared = {'cases': [], 'reasoning_effort': 'max', 'archive_sha256': 'zip', 'metadata_sha256': 'csv'}
    inputs = [('VULNERABLE', 'VULNERABLE', 'complete'), ('SAFE', 'SAFE', 'complete'),
              ('SAFE', 'VULNERABLE', 'complete'), ('VULNERABLE', 'SAFE', 'complete'),
              ('VULNERABLE', None, 'error'), ('SAFE', None, None)]
    for i, (truth, prediction, status) in enumerate(inputs):
        case = {'id': f'sample_{i:03d}', 'sample': i, 'source_sha256': str(i), 'truth': truth,
                'filename': f'sample_{i:03d}.c', 'prompt_tokens': 10}
        prepared['cases'].append(case)
        if status:
            evaluation.save(tmp_path / 'cases' / (case['id'] + '.json'), {
                **case, 'status': status, 'prediction': prediction, 'elapsed_seconds': 10,
                'usage': {'completion_tokens': 100}, 'output_tps_e2e': 10, 'decode_tps_estimate': 20})
    source, output = tmp_path / 'prepared.json', tmp_path / 'results.md'
    evaluation.save(source, prepared)
    evaluation.report(SimpleNamespace(prepared=source, output_dir=tmp_path, output=output))
    metrics = json.loads((tmp_path / 'metrics.json').read_text())
    assert [metrics[k] for k in ('TP', 'TN', 'FP', 'FN', 'errors', 'pending')] == [1, 1, 1, 1, 1, 1]
    assert metrics['precision'] == metrics['recall_completed'] == metrics['accuracy_completed'] == 0.5
    assert metrics['output_tps_e2e_weighted'] == 10
    assert 'Run incomplete' in output.read_text()
    assert 'sample_003.c' in output.read_text()


def test_stream_records_usage_and_completion(monkeypatch, tmp_path):
    events = [
        {'choices': [{'text': 'Analysis.\n', 'finish_reason': None}]},
        {'choices': [{'text': 'VERDICT: SAFE', 'finish_reason': None}]},
        {'choices': [{'text': '', 'finish_reason': 'stop'}]},
        {'choices': [], 'usage': {'prompt_tokens': 12, 'completion_tokens': 5}},
    ]
    stream = ''.join('data: ' + json.dumps(event) + '\n\n' for event in events) + 'data: [DONE]\n\n'
    monkeypatch.setattr(evaluation, 'urlopen', lambda *a, **kw: io.BytesIO(stream.encode()))
    result = evaluation.stream_review('http://local', {}, tmp_path / 'events.jsonl', 30)
    assert result['stream_done'] and result['finish_reason'] == 'stop'
    assert result['usage']['completion_tokens'] == 5
    assert evaluation.verdict(result['text']) == 'SAFE'
    assert len((tmp_path / 'events.jsonl').read_text().splitlines()) == 4
