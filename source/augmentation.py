import numpy as np


def mask_features(features, F: int, mf: int, T: int,
                  mt: int = None, ratio_t: float = None):
    """ SpecAugment: A Simple Data Augmentation Method. """
    time, channels = features.shape
    features = mask_frequencies(features, channels, F, mf)
    if ratio_t:     # Time dimension is chainging so ratio is more appropraite
        features = mask_time_ratio(features, time, T, ratio_t)
    else:
        features = mask_time(features, time, T, mt)
    return features


def mask_frequencies(features, channels: int, F: int, mf: int):
    means = features.mean(axis=0)
    for i in range(mf):
        f = np.random.random_integers(low=0, high=F)
        f0 = np.random.random_integers(low=0, high=channels-F)
        features[:, f0:f0+f] = means[f0:f0+f]   # should be zeros if normalized
    return features


def mask_time(features, time: int, T: int, mt: int):
    means = features.mean(axis=0)
    for i in range(mt):
        t = np.random.random_integers(low=0, high=T)
        t0 = np.random.random_integers(low=0, high=time-T)
        features[t0:t0+t, :] = means            # should be zeros if normalized
    return features


def mask_time_ratio(features, time: int, T: int, ratio_t: float):
    """ Mask overlap each other - samples differ of difficulty. """
    means = features.mean(axis=0)
    to_erase = time * ratio_t
    while to_erase > T/2:                       # larger than mean value
        t = np.random.random_integers(low=0, high=T)
        t0 = np.random.random_integers(low=0, high=time-T)
        features[t0:t0+t, :] = means            # should be zeros if normalized
        to_erase -= t
    return features
