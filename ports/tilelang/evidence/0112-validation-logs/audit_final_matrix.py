"""Verify final recorded artifacts without refreshing or modifying their reports."""
import hashlib,json,sys
from pathlib import Path
root,results,destination=map(Path,sys.argv[1:])
sys.path.insert(0,str(root/'ports/tilelang'))
from parity import evidence_inputs,evidence_state
from cpu_oracle import verify_probe_file
inputs=evidence_inputs(root)
features=json.loads((root/'ports/tilelang/parity.json').read_text())['features']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];totals={};files_checked=0
for t in ('c','metal','ascend'):
 totals[t]=0
 for f in features:
  op=f['operation'];d=results/('' if op=='solve' else op)/t
  path=d/'report.json';r=json.loads(path.read_text())
  state=evidence_state(root,r,inputs)
  assert state==('source_lowered' if t=='ascend' else 'executed'),(op,t,state)
  assert sha(d/'fixtures/layout.json')==r['layout_sha256']
  verify_probe_file(d/'cpu-oracle-noalias.ll',r['environment']['cpu_oracle'])
  for case in r['cases']:
   base=d/'fixtures'/('' if op=='solve' else op)/case['name']
   for name,h in case['files'].items():
    assert sha(base/name)==h,(op,t,case['name'],name)
    files_checked+=1
  oracle=r['environment']['cpu_oracle']
  extra=[('native_scene_uploads','native-inputs'),('native_restart_trajectory','native-trajectory')]
  for key,folder in extra:
   for name,h in oracle.get(key,{}).items():
    assert sha(d/'fixtures'/folder/name)==h,(op,t,key,name)
    files_checked+=1
  for name,h in oracle.get('production_pipeline',{}).get('files',{}).items():
   assert sha(d/'fixtures/production-reference'/name)==h,(op,t,'production',name)
   files_checked+=1
  totals[t]+=len(r['cases'])
  rows.append(dict(operation=op,target=t,state=state,cases=len(r['cases']),report_sha256=sha(path)))
 print(t,totals[t],'cases; artifact receipts verified',flush=True)
assert inputs==evidence_inputs(root)
assert not destination.exists()
destination.write_text(json.dumps(dict(schema=1,qualification='All registered operation artifacts and source fingerprints verified; runtime API and handoff acceptance remain separate gates.',inputs=inputs,operations=len(features),case_totals=totals,artifact_files_checked=files_checked,reports=rows,limitations=['Ascend source lowering only. No vendor binary, NPU execution, synchronization or performance qualification.']),indent=2)+'\n')
print('Wrote',destination)
