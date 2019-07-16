from source.deepspeech import DeepSpeech, Configuration, get_available_gpus
from scripts.run_extension import freeze, create_extended_model


def test_main():
    deepspeech = DeepSpeech.construct(config_path='configuration.yaml', alphabet_path='alphabet.txt')
    freeze(deepspeech.model)
    gpus = get_available_gpus()
    config = Configuration('configuration.yaml')
    extended_model = create_extended_model(deepspeech.model, config, is_gpu=len(gpus) > 0)
    assert len(deepspeech.model.layers) + len(config.data['extension']['layers']) == len(extended_model.layers)
    assert not extended_model.layers[9].trainable
    assert extended_model.layers[10].trainable

    optimizer = DeepSpeech.get_optimizer(**config.optimizer)
    loss = DeepSpeech.get_loss()
    gpus = get_available_gpus()
    deepspeech.compiled_model = DeepSpeech.compile_model(extended_model, optimizer, loss, gpus)
    assert len(deepspeech.compiled_model.layers) == len(extended_model.layers)
    assert not deepspeech.compiled_model.layers[9].trainable
    assert deepspeech.compiled_model.layers[10].trainable
