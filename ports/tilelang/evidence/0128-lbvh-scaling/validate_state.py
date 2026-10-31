"""Check the retained Balls comparison without a GPU or the original 1 GB inputs."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
layout = json.loads((HERE / 'after-layout.json').read_text())
assert layout['friction_model'] == 'twist'
before = json.loads((HERE / 'state-before.json').read_text())
after = json.loads((HERE / 'state-after.json').read_text())
arm = layout['twist']['arm_a'] // 128 * 4
count = layout['header']['normal'] // 128 * 4 + 3
checks = []
with zipfile.ZipFile(HERE / 'canonical-state.zip') as archive:
    for frame, (x, y) in enumerate(zip(before['state_sha256'], after['state_sha256'], strict=True)):
        differing = {name for name in x if x[name] != y[name]}
        assert differing <= {'OldIndices', 'OldLinks', 'OldTiles', 'canonical_tiles'}
        records = []
        for label, hashes in (('before', x), ('after', y)):
            data = archive.read(f'canonical-{label}/tiles-{frame}.bin')
            assert hashlib.sha256(data).hexdigest() == hashes['canonical_tiles']
            links = archive.read(f'canonical-{label}/links-{frame}.bin')
            assert hashlib.sha256(links).hexdigest() == hashes['canonical_links']
            records.append(np.frombuffer(data, np.uint32).reshape(-1, layout['tile_words']//32))
        a, b = records
        differences = np.argwhere(a != b)
        for row, offset in differences:
            point, component = divmod(int(offset) - arm, 4)
            assert 0 <= point < 4 and component < 3
            assert a[row, count] == b[row, count] and point >= int(a[row, count])
        checks.append({'frame': frame, 'exact_physical_and_spatial_hashes': len(x)-4,
                       'canonical_links_equal': x['canonical_links'] == y['canonical_links'],
                       'canonical_active_tile_words_equal': True,
                       'ignored_inactive_twist_arm_words': len(differences),
                       'pair_count': x['pair_count']})
assert checks == json.loads((HERE/'state-validation.json').read_text())['checks']
print('Six frames: exact physical state, spatial results, active tiles and warmstart caches.')
