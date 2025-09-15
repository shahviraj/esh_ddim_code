import matplotlib.pyplot as plt
import numpy as np
from load_deepmimo_datasets import load_deepmimo_datasets

# Helper to recursively extract all scalar los_status values (-1/1)
def extract_los_status(values):
    result = []
    if isinstance(values, (list, tuple, np.ndarray)):
        for v in values:
            result.extend(extract_los_status(v))
    else:
        # Try to convert to int (should be -1 or 1)
        try:
            val = int(values)
            if val in (-1, 1):
                result.append(val)
        except Exception:
            pass
    return result

# Load the DeepMIMO dataset (change path if needed)
data = load_deepmimo_datasets("../data/DeepMIMO_dataset", verbose=True)

# Collect all los_status values from all datasets
los_status_all = []
for dataset in data['datasets']:
    los = dataset.get('los_status', None)
    if los is not None:
        los_status_all.extend(extract_los_status(los))

los_status_all = np.array(los_status_all)

# Count occurrences of -1 and 1
unique, counts = np.unique(los_status_all, return_counts=True)
los_counts = dict(zip(unique, counts))

print("LoS Status Distribution:")
for k in [-1, 1]:
    count = los_counts.get(k, 0)
    print(f"  los_status = {k}: {count} samples")

# Plot
plt.figure(figsize=(6,4))
plt.bar([str(int(k)) for k in [-1, 1]], [los_counts.get(k, 0) for k in [-1, 1]], color=['red','green'])
plt.xlabel('LoS Status')
plt.ylabel('Count')
plt.title('Distribution of LoS Status in DeepMIMO Dataset')
plt.xticks([str(int(k)) for k in [-1, 1]])
plt.tight_layout()
plt.show()
