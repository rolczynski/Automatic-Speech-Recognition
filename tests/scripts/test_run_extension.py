from scripts.run_extension import freeze, create_extended_model
from source.deepspeech import DeepSpeech, Configuration, get_available_gpus
from source.utils import chdir

chdir(to='ROOT')


def test_main(deepspeech: DeepSpeech, config: Configuration):
    freeze(deepspeech.model)
    gpus = get_available_gpus()
    extended_model = create_extended_model(deepspeech.model, config, is_gpu=len(gpus) > 0)
    assert len(deepspeech.model.layers) + 1 + len(config.data['extension']['layers']) == len(extended_model.layers)
    *_, first_lstm, second_lstm, softmax = extended_model.layers
    assert not first_lstm.trainable
    assert second_lstm.trainable and softmax.trainable

    optimizer = DeepSpeech.get_optimizer(**config.optimizer)
    loss = DeepSpeech.get_loss()
    gpus = get_available_gpus()
    deepspeech.compiled_model = DeepSpeech.compile_model(extended_model, optimizer, loss, gpus)
    assert len(deepspeech.compiled_model.layers) == len(extended_model.layers)
    *_, first_lstm, second_lstm, softmax = deepspeech.compiled_model.layers
    assert not first_lstm.trainable
    assert second_lstm.trainable and softmax.trainable
