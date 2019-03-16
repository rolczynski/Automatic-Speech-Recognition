from typing import List
import pytest
import numpy as np
from metric import edit_distance, get_metrics


@pytest.fixture
def source() -> List[str]:
    return ['Ala', 'kópiła', 'kota']


@pytest.fixture
def destination() -> List[str]:
    return ['a', 'Ala', 'kupiła', 'kota']


def test_edit_distance(source: List[str], destination: List[str]):
    distance, edit_distance_matrix, backtrace = edit_distance(source, destination)
    assert distance == 2
    assert (edit_distance_matrix == np.array([[0, 1, 2, 3, 4],
                                             [1, 1, 1, 2, 3],
                                             [2, 2, 2, 2, 3],
                                             [3, 3, 3, 3, 2]])).all()
    assert all(backtrace[0] == np.array([(False, False, True, 0), (False, False, True, 0), (False, False, True, 0),
                                         (False, False, True, 0), (False, False, True, 0)],
                                        dtype=[('del', bool), ('sub', bool), ('ins', bool), ('cost', int)]))


@pytest.fixture
def sources() -> List[str]:
    return ['Ala kópiła kota']


@pytest.fixture
def destinations() -> List[str]:
    return ['a Ala kupiła kota']


def test_get_metrics(sources: List[str], destinations: List[str]):
    metrics_generator = get_metrics(sources, destinations)
    metrics = list(metrics_generator)
    assert len(metrics) == 1
    metric, = metrics
    assert metric.transcript == 'a Ala kupiła kota'
    assert metric.prediction == 'Ala kópiła kota'
    assert isinstance(metric.cer, np.float)
    assert np.isclose(metric.cer, 0.17, rtol=.05)
    assert isinstance(metric.wer, np.float)
    assert metric.wer == 0.5
