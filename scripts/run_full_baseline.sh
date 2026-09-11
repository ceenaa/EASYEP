#!/bin/bash
set -euo pipefail
cd /workspace/prunungdsk4
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/FreeToken/python:$PWD/scripts"
python -u - <<'PY'
import json, time
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen
from evaluate_full_model import baseline_server, get_json, save
from freetoken.pruning.workflow import tokenizer_tools

server='http://127.0.0.1:1919'
prepared=json.loads(Path('results/primevul-v03-prepared.json').read_text())
for attempt in range(720):
    try:
        health=get_json(server,'/health')
    except OSError:
        health={'status':'unreachable'}
    if health.get('status')=='error':
        raise RuntimeError(str(health))
    if health.get('status')=='ok':
        break
    if attempt % 12 == 0:
        print('Waiting for model:',health,flush=True)
    time.sleep(5)
else:
    raise TimeoutError('Model was not ready within one hour')
model,metadata=baseline_server(server,prepared['checkpoint'])
_,manager=tokenizer_tools('models/DeepSeek-V4-Flash-0731')
prompt=manager.render_prompt(SimpleNamespace(text=[{'role':'user','content':'What is 2 + 2? Reply with only the number.'}],tools=None,chat_template_kwargs={'thinking_mode':'chat'}))
payload={'model':model,'prompt':prompt,'max_tokens':32,'temperature':0,'top_k':1,'top_p':1,'stream':False}
started=time.monotonic()
with urlopen(Request(server+'/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'}),timeout=1800) as response:
    result=json.load(response)
save('results/full-model-smoke-v03.json',{'server':metadata,'response':result,'elapsed_seconds':time.monotonic()-started})
assert result['choices'][0]['finish_reason']=='stop',result
assert result['choices'][0]['text'].strip()=='4',result
print('Full-model arithmetic smoke check passed',flush=True)
PY
exec python -u scripts/evaluate_full_model.py run --prepared results/primevul-v03-prepared.json --output-dir results/full-baseline-v03 --max-tokens 57344 --max-seq-len 65536 --resume
