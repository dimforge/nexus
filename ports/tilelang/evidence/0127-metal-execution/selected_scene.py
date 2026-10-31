import argparse,sys
from pathlib import Path
root=Path('/Users/sebcrozet/work/worktrees/nexus/tilelang-parity/ports/tilelang')
sys.path.insert(0,str(root))
from cpu_oracle import export_fixtures
from runtime_replay import BUFFERS
from runtime_scene_validation import validate_case,check_outputs
from scene_kernels import make_program
from stage_replay import run_stage
work=Path('/private/tmp/nexus-tilelang-execution-probe-20261031')
args=argparse.Namespace(operation='runtime_scene',target='metal',output=work/'selected-scene',compile_only=False,selected_cases=['native-upload'])
run_stage(args,oracle='runtime-scenes',exporter=lambda d: export_fixtures('runtime-scenes',d,(work/'selected-native-upload',8,64,'execute','reduced')),buffers=BUFFERS,make_kernel=None,make_program=make_program,validate_case=validate_case,check_outputs=check_outputs,launch=None)
