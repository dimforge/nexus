import json, os, signal, subprocess, time
from pathlib import Path
root=Path('/Users/sebcrozet/work/worktrees/nexus/tilelang-parity'); port=root/'ports/tilelang'; results=port/'results-testbed-0112'; base=Path('/private/tmp/nexus-frame-runtime')
features=json.loads((port/'parity.json').read_text())['features']
first=json.loads((base/'testbed0112-job-order.json').read_text())[:6]
retired=False; last=None; deadline=time.monotonic()+21600
while True:
 states=[]; by_key={}; failed=[]
 for target in ('c','metal','ascend'):
  for f in features:
   op=f['operation']; p=results/('' if op=='solve' else op)/target/'report.json'
   try: report=json.loads(p.read_text())
   except (FileNotFoundError,json.JSONDecodeError): states.append('pending');continue
   state=report.get('status'); states.append(state);by_key[(target,op)]=state
   if state not in ('preparing','passed'):failed.append((op,target,state))
 if failed: raise SystemExit(f'Failed reports: {failed}')
 if not retired and all(by_key.get((j['target'],j['operation']))=='passed' for j in first):
  command=subprocess.run(['ps','-p','62183','-o','command='],capture_output=True,text=True).stdout
  if 'testbed0112-matrix.py' in command: os.kill(62183,signal.SIGKILL)
  retired=True;print('Six original runtime jobs passed; retired the stopped scheduler (no active child work remains).',flush=True)
 count=states.count('passed')
 if count!=last:print(f'{count}/456 reports passed',flush=True);last=count
 if count==456:break
 if time.monotonic()>deadline:raise SystemExit('Matrix validation timeout; inspect current workers')
 time.sleep(30)
subprocess.run(['python3',str(base/'audit_final_matrix.py'),str(root),str(results),str(port/'evidence/0112-testbed-matrix-checkpoint.json')],cwd=root,check=True)
env=os.environ.copy();env.update(NEXUS_TILELANG_RESULTS=str(results),TILELANG_CACHE_DIR='/private/tmp/nexus-tilelang-probe-cache',TVM_FFI_CACHE_DIR='/private/tmp/nexus-tvm-ffi-cache')
with (base/'testbed0112-all-python.log').open('w') as log:
 subprocess.run(['/private/tmp/nexus-tilelang-probe-20261029/bin/python','-m','unittest','discover','-s','ports/tilelang/tests','-v'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Full current-fixture Python suite passed',flush=True)
for args,name in [(['reconcile','--change','ports/tilelang/changes/0112-testbed-integration.json','--evidence-dir',str(results),'--json'],'reconciliation'),(['check','--require','c','--require','metal','--release','--json'],'release-check')]:
 with (base/f'testbed0112-{name}.json').open('w') as log:
  subprocess.run(['python3','ports/tilelang/parity.py',*args],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Current C/Metal execution, Ascend source lowering, regressions and parity check passed',flush=True)
