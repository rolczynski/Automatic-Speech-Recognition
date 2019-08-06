import os
import numpy as np
from scripts import run
is_same = lambda A, B: all(np.array_equal(a, b) for a, b in zip(A, B))
is_close = lambda A, B: all(np.allclose(a, b, atol=1e-04) for a, b in zip(A, B))


def test_deepspeech_integration():
    base_deepspeech = run.setup_deepspeech('tests/models/base/configuration.yaml',
                                           'tests/models/base/alphabet.txt')
    base_weights = 'weights.hdf5'
    base_deepspeech.save(base_weights)

    extended_deepspeech = run.setup_deepspeech('tests/models/extended/configuration.yaml',
                                               'tests/models/extended/alphabet.txt')
    assert all(not extended_deepspeech.model.get_layer(name).trainable for name in ['base_1', 'base_2', 'base_3'])
    assert all(not is_same(base_deepspeech.model.get_layer(name).get_weights(),
                           extended_deepspeech.model.get_layer(name).get_weights())
               for name in ['base_1', 'base_2', 'base_3'])

    extended_deepspeech = run.setup_deepspeech('tests/models/extended/configuration.yaml',
                                               'tests/models/extended/alphabet.txt',
                                               pretrained_weights=base_weights)
    assert all(not extended_deepspeech.model.get_layer(name).trainable for name in ['base_1', 'base_2', 'base_3'])
    assert all(is_same(base_deepspeech.model.get_layer(name).get_weights(),
                       extended_deepspeech.model.get_layer(name).get_weights())
               for name in ['base_1', 'base_2', 'base_3'])

    for i in range(10):                                                            # Dummy training (10 epochs / 10 batch_size)
        X = np.random.rand(10, 100, 80)
        y = np.random.randint(0, 35, size=[10, 20], dtype=np.int32)
        extended_deepspeech.parallel_model.train_on_batch(X, y)

    assert is_close(extended_deepspeech.model.predict(X), extended_deepspeech.parallel_model.predict(X)), \
        "The results are the same for model and compiled parallel model."
    os.remove(base_weights)


def test_create_parser():
    parser = run.create_parser()
    args = parser.parse_args("""
    --model_dir test
    --train train_features.hdf5
    --dev dev_features.hdf5
    --source from_prepared_features
    --source_distributed
    --batch_size 256
    --epochs 15
    --pretrained_weights model/weights.hdf5
    """.split())
    assert all(isinstance(param, int)
               for param in [args.epochs, args.log_level, args.batch_size])
    assert isinstance(args.source_distributed, bool)
