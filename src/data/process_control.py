import numpy as np


data = np.load('../../data/control_signals/--GVEgZn_TI/shot_--GVEgZn_TI_shot_0000_controls.npz', allow_pickle=True)


print("Arrays in file:", data.files)
# Output: ['depth', 'edges', 'flow', 'metadata']


for key in data.files:
    arr = data[key]
    print(f"{key}: shape={arr.shape}, dtype={arr.dtype}, size={arr.nbytes / 1024:.2f} KB")


data.close()