"""Validate snippets using the locally built nexus3d extension (no mocks).

Pass --gpu to also execute the snippets needing a headless viewer/backend.
"""
import ast
import pathlib
import sys

for source in pathlib.Path(__file__).parent.glob("*.py"):
    ast.parse(source.read_text(), filename=str(source))

import getting_started  # Verify all imports without opening a window.
import guide
from nexus3d import NexusState, NexusViewer

state = NexusState()
guide.bodies(state)
guide.colliders(state)
guide.joints(state)
guide.collision_groups()
guide.parameters(state)
guide.friction_model(state)
assert state.rbd_solver_params()["friction_model"] == "simplified"
guide.coupling(state)
assert state.counts().rigid_bodies == 5
assert guide.batches().counts().num_environments == 4
assert len(guide.material_models()) == 2
assert len(guide.boundaries()) == 3
state = NexusState()
guide.multibody(state)
assert state.counts().multibodies == 1
print("Python imports and scene-building snippets passed against nexus3d.")

if "--gpu" in sys.argv:
    viewer = NexusViewer(width=64, height=64, headless=True)
    viewer.init_backend()
    state = guide.batches()
    guide.determinism(viewer, state)
    assert state.deterministic()
    guide.lifecycle(viewer, state)
    assert state.rbd_num_batches() == 4
    state = guide.insertion(viewer)
    guide.lifecycle(viewer, state)
    state = NexusState()
    chunk = guide.particles(viewer, state)
    guide.coupling(state)
    guide.lifecycle(viewer, state)
    guide.remove_particles(viewer, state, chunk)
    assert state.counts().particles == 0
    state = NexusState()
    guide.multibody(state)
    guide.lifecycle(viewer, state)
    guide.motors(viewer, state)
    guide.lifecycle(viewer, state)
    positions = guide.readback(viewer, state)
    assert positions.shape == (2, 3)
    print("Python backend snippets passed.")
