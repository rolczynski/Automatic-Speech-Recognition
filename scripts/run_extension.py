import os
import numpy as np
from tensorflow import set_random_seed

from scripts.run import parse_arguments, create_generators
from source.deepspeech import DeepSpeech, Configuration, get_available_gpus
from source.model import CuDNNLSTM, Bidirectional, Dense, TimeDistributed, Model, ReLU, LSTM, BatchNormalization
from source.utils import chdir, create_logger


def freeze(model: Model):
    for layer in model.layers:
        layer.trainable = False


def create_extended_model(model: Model, configuration: Configuration, is_gpu: bool, random_state=1) -> Model:
    np.random.seed(random_state)
    set_random_seed(random_state)

    constructors = {
        'BatchNormalization': lambda params: BatchNormalization(**params),
        'Dense': lambda params: TimeDistributed(Dense(**params)),
        'LSTM': lambda params: Bidirectional(CuDNNLSTM(**params) if is_gpu else
                                             LSTM(activation='tanh', recurrent_activation='sigmoid', **params),
                                             merge_mode='sum'),
        'ReLU': lambda params: ReLU(**params)
    }

    input_tensor = model.inputs[0]
    x = model.layers[-2].output     # without softmax layer

    layers = configuration.data['extension']['layers']
    for params in layers:
        name = params.pop('name')
        constructor = constructors[name]
        x = constructor(params)(x)

    *_, output_dim = model.layers[-1].output_shape
    output_tensor = TimeDistributed(Dense(units=output_dim, activation='softmax'))(x)
    model = Model(input_tensor, output_tensor, name='DeepSpeech')
    return model


def main(args):
    deepspeech = DeepSpeech.construct(config_path=CONFIG_PATH, alphabet_path=ALPHABET_PATH)
    if args.pretrained_weights:
        deepspeech.load(args.pretrained_weights)

    freeze(deepspeech.model)
    gpus = get_available_gpus()
    config = Configuration(CONFIG_PATH)
    extended_model = create_extended_model(deepspeech.model, config, is_gpu=len(gpus) > 0)

    optimizer = DeepSpeech.get_optimizer(**config.optimizer)
    loss = DeepSpeech.get_loss()
    gpus = get_available_gpus()
    deepspeech.model = extended_model
    deepspeech.compiled_model = DeepSpeech.compile_model(extended_model, optimizer, loss, gpus)

    train_generator, dev_generator = create_generators(deepspeech, args)
    deepspeech.fit(train_generator, dev_generator, epochs=args.epochs, shuffle=False)
    deepspeech.save(WEIGHTS_PATH)


if __name__ == "__main__":
    chdir(to='ROOT')
    ARGUMENTS = parse_arguments()

    LOG_PATH = os.path.join(ARGUMENTS.model_dir, 'training.log')
    CONFIG_PATH = os.path.join(ARGUMENTS.model_dir, 'configuration.yaml')
    ALPHABET_PATH = os.path.join(ARGUMENTS.model_dir, 'alphabet.txt')
    WEIGHTS_PATH = os.path.join(ARGUMENTS.model_dir, 'weights.hdf5')

    logger = create_logger(LOG_PATH, level=ARGUMENTS.log_level, name='deepspeech')
    logger.info(f'Arguments: \n{ARGUMENTS}')

    main(ARGUMENTS)
