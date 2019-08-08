from typing import List
import pytest
import numpy as np
from matplotlib import pyplot as plt
import augmentation
from source.audio import FeaturesExtractor
from source.utils import chdir
chdir(to='ROOT')
debugging = False
# debugging = True
np.random.seed(123)


def plot(features):
    if debugging:
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


def test_mask_time_stripes(features: np.ndarray):
    time, channels = features.shape
    means = features.mean(axis=0)
    masked = augmentation.mask_time_stripes(np.copy(features), means, time, T_range=[5, 10], ratio=0.3, space=5)
    plot(masked)
    ratio = sum(np.array_equal(masked[t, :], means) for t in range(time)) / time
    assert np.isclose(ratio, 0.3, atol=0.1)
    masked = augmentation.mask_time_stripes(np.copy(features), means, time, T_range=[5, 50], ratio=0.5, space=150)     # Avoid infinite loop
    plot(masked)


def test_mask_features(features: np.ndarray):
    masked = augmentation.mask_features(np.copy(features), F=20, mf=2)
    plot(masked)
    masked = augmentation.mask_features(np.copy(features), Tmax=40, ratio_t=0.3)
    plot(masked)
