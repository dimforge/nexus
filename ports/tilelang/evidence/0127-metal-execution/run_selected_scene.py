import json,os,signal,subprocess,time
from pathlib import Path
root=Path('/Users/sebcrozet/work/worktrees/nexus/tilelang-parity')
work=Path('/private/tmp/nexus-tilelang-execution-probe-20261031')
cmd=['/private/tmp/nexus-tilelang-probe-20261029/bin/python',str(work/'selected_scene.py')]
env=dict(os.environ,TILELANG_CACHE_DIR='/private/tmp/nexus-tilelang-probe-cache',TVM_FFI_CACHE_DIR='/private/tmp/nexus-tvm-ffi-cache')
start=time.monotonic()
with (work/'selected-scene.log').open('w') as log:
 p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try: code=p.wait(timeout=290)
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGKILL);p.wait();code=124
receipt=dict(command=cmd,seconds=time.monotonic()-start,code=code)
(work/'selected-scene-timing.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(receipt)
raise SystemExit(code)
