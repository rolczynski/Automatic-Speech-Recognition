import pytest
import h5py
import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
from scripts.features import extract_features, divide, concatenate, generate_samples_from


@pytest.fixture
def audio_path() -> str:
    return 'tests/data/audio.csv'


@pytest.fixture
def segmented_path() -> str:
    return 'tests/data/segmented.csv'


@pytest.fixture
def max_words() -> int:
    return 7


@pytest.fixture
def mfcc_params() -> dict:
    return dict(winlen=0.032,
                winstep=0.02,
                numcep=26,
                winfunc=np.hamming)


@pytest.fixture
def audio(audio_path: str) -> pd.DataFrame:
    return pd.read_csv(audio_path, index_col='id')


@pytest.fixture
def fs() -> int:
    return 16000


@pytest.fixture
def segmented(segmented_path: str) -> pd.DataFrame:
    segmented_data = pd.read_csv(segmented_path)
    return segmented_data.groupby('audio_id')


@pytest.fixture
def audio_id() -> int:
    return 0    # Test first audio sample


@pytest.fixture
def file_path(audio_id: int, audio: pd.DataFrame) -> str:
    return audio.loc[audio_id, 'path']


@pytest.fixture
def audio_data(file_path: str) -> np.ndarray:
    _, audio = wav.read(file_path)
    return audio


@pytest.fixture
def segmented_audio(audio_id: int, segmented: pd.DataFrame) -> pd.DataFrame:
    return segmented.get_group(audio_id)


def test_divide():
    pass


def test_concatenate():
    pass


@pytest.fixture
def segmented_phrases(segmented_audio: pd.DataFrame, max_words: int) -> pd.DataFrame:
    phrase_groups = divide(segmented_audio, max_words)
    segmented_phrases = concatenate(phrase_groups)
    assert all(segmented_phrases.length > 0)
    assert all((segmented_phrases.end - segmented_phrases.start) > 0)
    return segmented_phrases


def test_generate_samples_from(segmented_phrases: pd.DataFrame, audio_data: np.ndarray, fs: int, mfcc_params: dict):
    samples_generator = generate_samples_from(segmented_phrases, audio_data, fs, mfcc_params)
    samples = list(samples_generator)
    correct_samples = [sample for sample in samples if sample]
    assert len(correct_samples) == 4
    firs_sample, *_ = correct_samples
    assert firs_sample.size == 71680
    assert firs_sample.features.shape == (224, 26)


@pytest.fixture
def store_path() -> str:
    return 'tests/data/features.hdf5'


def test_extract_features(store_path: str, audio_path: str, segmented_path: str, max_words: int, mfcc_params: dict):
    extract_features(store_path, audio_path, segmented_path, max_words, mfcc_params)
    with pd.HDFStore(store_path, mode='r') as store:
        info = store['info']
        assert list(info.columns) == ['winlen', 'winstep', 'numcep', 'winfunc']
        references = store['references']
        assert list(references.columns) == ['path', 'size', 'transcript']
        assert references.shape == (12, 3)
        assert all(np.diff(references['size']) > 0)

    random_record = references.sample()
    record_path, = random_record.path
    with h5py.File(store_path, mode='r') as store:
        sample = store[record_path]
        assert sample.dtype == np.float
        assert sample.ndim == 2
