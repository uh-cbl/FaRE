import numpy as np


def compute_mean(features):
    if not isinstance(features, np.ndarray):
        features = np.concatenate(features)

    embed_size = features.shape[-1]
    if len(features.shape) != 2:
        features = features.reshape((-1, embed_size))

    return features.mean(axis=0).reshape((1, -1))

