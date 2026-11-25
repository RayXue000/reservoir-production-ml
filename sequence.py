"""Data-independent form of the project's next-step sliding-window construction."""
import numpy as np


def make_windows(features, target_index, timestep=2):
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or timestep < 1 or len(values) <= timestep:
        raise ValueError("Use [time, features] data with more rows than the window")
    if not 0 <= target_index < values.shape[1]:
        raise ValueError("target_index must identify one input feature")
    x = np.stack([values[i:i + timestep] for i in range(len(values) - timestep)])
    y = values[timestep:, target_index:target_index + 1]
    return x, y
