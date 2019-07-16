from typing import List

import pytest
import numpy as np
from matplotlib import pyplot as plt
from source.audio import FeaturesExtractor
import augmentation
from source.utils import chdir
chdir(to='ROOT')


def plot(features):
    fix, ax = plt.subplots(figsize=(15, 5))
    ax.imshow(features.T)
    plt.show()


@pytest.fixture
def features(audio_file_paths: List[str]):
    feature_extractor = FeaturesExtractor(
        winlen=0.025,
        winstep=0.01,
        nfilt=80,
        winfunc='hamming'
    )
    feat = feature_extractor.get_features(
        files=[audio_file_paths[0]]
    )[0]
    return (feat-feat.mean(axis=0)) / feat.std(axis=0)


def test_mask_features(features: np.ndarray):
    features = features[:500]
    masked = augmentation.mask_features(np.copy(features), F=20, mf=2, T=10, ratio_t=0.1)
    plot(masked)
    masked = augmentation.mask_features(np.copy(features), F=20, mf=2, T=20, ratio_t=0.1)
    plot(masked)
    masked = augmentation.mask_features(np.copy(features), F=20, mf=2, T=10, ratio_t=0.2)
    plot(masked)
    masked = augmentation.mask_features(np.copy(features), F=20, mf=2, T=20, ratio_t=0.2)
    plot(masked)
    masked = augmentation.mask_features(np.copy(features), F=20, mf=2, T=10, ratio_t=0.3)
    plot(masked)
    masked = augmentation.mask_features(np.copy(features), F=20, mf=2, T=20, ratio_t=0.3)
    plot(masked)


def test_mask_frequencies(features: np.ndarray):
    features = features[: 300]        # Trim time dimension
    time, channels = features.shape
    augmentation.mask_frequencies(features, channels, F=20, mf=2)
    plot(features)
    augmentation.mask_time(features, time, T=20, mt=10)
    plot(features)
