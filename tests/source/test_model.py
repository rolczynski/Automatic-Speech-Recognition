import os
import numpy as np
import deepspeech
from deepspeech import DeepSpeech
from source.model import deepspeech_custom
from source.utils import chdir

is_same = lambda A, B: all(np.array_equal(a, b) for a, b in zip(A, B))
is_close = lambda A, B: all(np.allclose(a, b, atol=1e-04) for a, b in zip(A, B))
chdir(to='ROOT')


def test_trainable():
    gpus = deepspeech.get_available_gpus()      # Support both Multi and Single-GPU tests
    base_configuration = DeepSpeech.get_configuration('tests/models/base/configuration.yaml')
    base_configuration.model.pop('name')
    base_model = deepspeech_custom(is_gpu=len(gpus) > 0, **base_configuration.model)
    fname = 'weights.hdf5'
    base_model.save_weights(fname)

    extended_configuration = DeepSpeech.get_configuration('tests/models/extended/configuration.yaml')
    extended_configuration.model.pop('name')
    extended_model = deepspeech_custom(is_gpu=len(gpus) > 0, **extended_configuration.model)
    weights_before_training = extended_model.get_weights()

    assert all(not extended_model.get_layer(name).trainable for name in ['base_1', 'base_2', 'base_3'])
    assert all(extended_model.get_layer(name).trainable for name in ['extension_1', 'extension_2'])
    assert all(is_same(base_model.get_layer(name).get_weights(), extended_model.get_layer(name).get_weights())
               for name in ['base_1', 'base_2', 'base_3'])

    loss = DeepSpeech.get_loss()
    optimizer = DeepSpeech.get_optimizer(**extended_configuration.optimizer)
    parallel_model = DeepSpeech.distribute_model(extended_model, gpus)
    DeepSpeech.compile_model(parallel_model, optimizer, loss)
    parallel_model.load_weights(fname, by_name=True)

    for i in range(10):                                                            # Dummy training (10 epochs / 10 batch_size)
        X = np.random.rand(10, 100, 80)
        y = np.random.randint(0, 35, size=[10, 20], dtype=np.int32)
        parallel_model.train_on_batch(X, y)

    assert all(is_same(base_model.get_layer(name).get_weights(), extended_model.get_layer(name).get_weights())
               for name in ['base_1', 'base_2', 'base_3']), "Freezed layers have to be unchangeable."
    assert not is_same(weights_before_training, extended_model.get_weights()), "The base model updates weights."
    assert is_close(extended_model.predict(X), parallel_model.predict(X)), "The results are the same (compiled model)."
    # assert is_same(extended_model.get_weights(), compiled_model.get_weights())    # Weights can not be compared (order changed)
    os.remove(fname)
