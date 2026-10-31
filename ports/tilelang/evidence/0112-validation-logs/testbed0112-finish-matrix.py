"""Finish fresh parity jobs with at most twelve active replay processes."""
import concurrent.futures as cf
import json, os, signal, subprocess, sys, threading, time
from pathlib import Path
root=Path('/Users/sebcrozet/work/worktrees/nexus/tilelang-parity'); port=root/'ports/tilelang'; out=port/'results-testbed-0112'; base=Path('/private/tmp/nexus-frame-runtime')
sys.path.insert(0,str(port))
from matrix import features,command
old_pid=69776
old_cmd=subprocess.check_output(['ps','-p',str(old_pid),'-o','command='],text=True)
assert 'testbed0112-remaining.py' in old_cmd
os.kill(old_pid,signal.SIGSTOP)
try:
    processes=subprocess.check_output(['ps','-axo','pid=,ppid=,command='],text=True)
    active=set()
    for line in processes.splitlines():
        fields=line.split()
        if len(fields)<3 or fields[1]!=str(old_pid): continue
        if '--target' not in fields or '--operation' not in fields:
            raise RuntimeError(f'Unexpected child of old scheduler: {line}')
        active.add((fields[fields.index('--target')+1],fields[fields.index('--operation')+1]))
    def status(target,op):
        p=out/('' if op=='solve' else op)/target/'report.json'
        try:return json.loads(p.read_text()).get('status')
        except (FileNotFoundError,json.JSONDecodeError):return None
    rows={r['operation']:r for r in features()}
    schedule=json.loads((base/'testbed0112-job-order.json').read_text())
    jobs=[j for j in schedule if (j['target'],j['operation']) not in active and status(j['target'],j['operation'])!='passed']
    assert len(active)<=8
    (base/'testbed0112-finish-schedule.json').write_text(json.dumps({'continued_children':sorted(active),'new_jobs':jobs,'maximum_concurrent_replay_processes':12},indent=2)+'\n')
except BaseException:
    os.kill(old_pid,signal.SIGCONT)
    raise
print(f'Continuing {len(active)} existing jobs and {len(jobs)} pending fresh jobs; maximum 12 replay processes; no existing result is rerun.',flush=True)
slots=threading.Semaphore(12-len(active));errors=[]
def retire():
    pending=set(active)
    while pending:
        for key in list(pending):
            state=status(*key)
            if state=='passed':
                pending.remove(key);slots.release();print('CONTINUED PASSED',*key,flush=True)
            elif state not in (None,'preparing'):
                pending.remove(key);slots.release();errors.append(f'continued {key}: {state}');print('CONTINUED FAILED',*key,state,flush=True)
        if pending:time.sleep(5)
    # All direct child replay commands have written terminal reports. The
    # stopped parent must not resume its old duplicate task queue.
    os.kill(old_pid,signal.SIGKILL)
    print('Retired the stopped eight-worker scheduler after all of its active reports completed.',flush=True)
watch=threading.Thread(target=retire);watch.start()
def run(job):
    target=job['target'];op=job['operation'];row=rows[op];ascend=target=='ascend'
    with slots:
        env=os.environ.copy();env['TILELANG_CACHE_DIR']='/private/tmp/nexus-tilelang-ascend-cache' if ascend else '/private/tmp/nexus-tilelang-probe-cache';env['TVM_FFI_CACHE_DIR']='/private/tmp/nexus-tvm-ffi-cache'
        cmd=command(row,target,out,ascend);cmd[0]='/private/tmp/nexus-tilelang-ascend-20261029/bin/python' if ascend else '/private/tmp/nexus-tilelang-probe-20261029/bin/python'
        with (base/f'testbed0112-{target}-{op}.log').open('w') as log:
            subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    return target,op
with cf.ThreadPoolExecutor(max_workers=12) as pool:
    futures={pool.submit(run,j):j for j in jobs}
    for f in cf.as_completed(futures):
        try:print('PASSED',*f.result(),flush=True)
        except Exception as e:errors.append(str(e));print('FAILED',futures[f]['target'],futures[f]['operation'],str(e),flush=True)
watch.join()
if errors:raise SystemExit('\n'.join(errors))
print('All continued and remaining matrix jobs passed.',flush=True)
