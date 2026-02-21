
import numpy as np
from pathlib import Path

control_dir = Path('/mnt/d1/controllable-generation/encoded_controls')
files = list(control_dir.rglob('*_encoded.npz'))
print(f'Total files: {len(files)}')

bad = []
for i, f in enumerate(files):
    if i % 50 == 0:
        print(f'  Checking {i}/{len(files)}...')
    try:
        data = np.load(f)
        for k in data.keys():
            arr = data[k]
            if np.isnan(arr).any() or np.isinf(arr).any():
                bad.append((str(f), k, 'nan/inf'))
            elif np.abs(arr).max() > 100:
                bad.append((str(f), k, f'large values: {np.abs(arr).max():.1f}'))
    except Exception as e:
        bad.append((str(f), 'load_error', str(e)))

print(f'\nBad files: {len(bad)}')
for b in bad[:20]:
    print(b)
