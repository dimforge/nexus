import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
root=Path('/Users/sebcrozet/work/worktrees/nexus/tilelang-parity');port=root/'ports/tilelang';base=Path('/private/tmp/nexus-frame-runtime');out=port/'results-testbed-0112';archive=port/'evidence/0112-testbed-matrix-retries'
archive.mkdir(exist_ok=False)
sys.path.insert(0,str(port));from matrix import command,features
rows={r['operation']:r for r in features()}
failures=[]
for op in ('luby_reset','segment_projection'):
 report=out/op/'metal/report.json';r=json.loads(report.read_text());assert r['status']=='failed'
 log=base/f'testbed0112-metal-{op}.log'
 shutil.copy2(report,archive/f'{op}.failed-report.json');shutil.copy2(log,archive/f'{op}.failed.log')
 failures.append({'operation':op,'target':'metal','error':r['error'],'failed_case':r['failed_case']})
env=os.environ.copy();env.update(TILELANG_CACHE_DIR='/private/tmp/nexus-tilelang-probe-cache',TVM_FFI_CACHE_DIR='/private/tmp/nexus-tvm-ffi-cache')
for op in ('luby_reset','segment_projection'):
 cmd=command(rows[op],'metal',out);cmd[0]='/private/tmp/nexus-tilelang-probe-20261029/bin/python'
 print('Fresh serial retry',op,flush=True)
 with (archive/f'{op}.serial-retry.log').open('w') as log:subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 r=json.loads((out/op/'metal/report.json').read_text());assert r['status']=='passed';print('PASSED',op,len(r['cases']),'cases',flush=True)
receipt={'schema':1,'qualification':'Initial parallel runner failures are preserved; both operations were rerun in full from fresh Rust fixtures without concurrent Cargo builds. No solver code, case or tolerance changed.','failures':failures,'diagnosis':'All replay builds share target/debug/native-replay. One launch observed that file missing during concurrent build activity; another process ended with SIGKILL and no output. Concurrent executable replacement is suspected for the launch failures, but the SIGKILL cause is not proven. The stock matrix.py runs each shard serially.','resolution':'Serial full reruns, followed by the complete artifact audit and regression gates. Future parallel local sharding should isolate build/output workspaces or stabilize executables before launching concurrent replays.','artifacts':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(archive.iterdir())}}
(archive/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
change={'id':'0116-testbed-validation-retries','features':[rows[o]['id'] for o in ('luby_reset','segment_projection')],'kinds':['test','tooling'],'disposition':'behavior_unchanged','summary':'Record two process-launch failures during the temporary parallel testbed parity refresh and retain complete serial reruns without changing solver sources or numerical contracts.','regressions':['Fresh full Metal luby_reset suite, run without concurrent Cargo builds.','Fresh full Metal segment_projection suite, run without concurrent Cargo builds.','Full 456-report current-source artifact audit and complete Python regression suite.'],'differences':['The first runs failed before a result could be compared: one missing shared native-replay executable and one SIGKILL with no output.','Executable replacement under concurrent Cargo builds is suspected; the SIGKILL cause is unconfirmed. Stock matrix.py is serial per shard.','Original failure reports and logs are preserved in evidence/0112-testbed-matrix-retries. Serial reruns must pass before acceptance.','No arithmetic, cases, tolerances, runtime implementation or shader mapping changes.']}
(port/'changes/0116-testbed-validation-retries.json').write_text(json.dumps(change,indent=2)+'\n')
print('Serial reruns and failure provenance retained.',flush=True)
