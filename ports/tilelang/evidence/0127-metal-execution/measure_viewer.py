import json,os,re,signal,subprocess,sys,time
from pathlib import Path
root=Path('/Users/sebcrozet/work/worktrees/nexus/tilelang-parity')
work=Path('/private/tmp/nexus-tilelang-execution-probe-20261031')
mode=sys.argv[1]
env=dict(os.environ,NEXUS_TILELANG_PYTHON='/private/tmp/nexus-tilelang-probe-20261029/bin/python',NEXUS_TILELANG_KERNEL_CACHE='/private/tmp/nexus-tilelang-preparation-20261031/testbed-final-cache',NEXUS_TILELANG_WORKDIR=str(work/'testbed-work'),TILELANG_CACHE_DIR='/private/tmp/nexus-tilelang-probe-cache',TVM_FFI_CACHE_DIR='/private/tmp/nexus-tvm-ffi-cache')
cmd=[str(root/'target/debug/all_examples3'),'--example','tileLangComparison','--tilelang','metal','--run','--headless','--snapshot','14',str(work/(mode+'.png'))]
if '--compare' in sys.argv:cmd.append('--compare')
start=time.monotonic()
with (work/(mode+'.log')).open('w') as log:
 p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:code=p.wait(timeout=285)
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGKILL);p.wait();raise
receipt={'command':cmd,'elapsed_seconds':time.monotonic()-start,'code':code}
(work/(mode+'-timing.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print(receipt)
for line in (work/(mode+'.log')).read_text().splitlines():
 if line.startswith(('TileLang ready','TileLang testbed result')):print(line)
raise SystemExit(code)
