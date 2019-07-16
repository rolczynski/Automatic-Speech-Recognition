import os
import argparse
import numpy as np
from tensorflow import set_random_seed
from source.deepspeech import DeepSpeech, Configuration, get_available_gpus
from source.model import CuDNNLSTM, Bidirectional, Dense, TimeDistributed, Model, ReLU, LSTM, BatchNormalization
from source.utils import chdir, create_logger


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help='DeepSpeech model directory')
    parser.add_argument('--train', required=True, help='Train source (csv/hdf5 file)')
    parser.add_argument('--dev', required=True, help='Dev source (csv/hdf5 file)')
    parser.add_argument('--source', choices=['from_audio_files', 'from_prepared_features'], required=True)
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (int)')
    parser.add_argument('--epochs', type=int, default=25, help='Epochs during training')
    parser.add_argument('--shuffle_after_epoch', type=int, default=1, help='Shuffle generator indices after epoch')
    parser.add_argument('--pretrained_weights', help='Use weights from the pretrained model')
    parser.add_argument('--mask', dest='mask', action='store_true', help='Mask features during training')
    parser.add_argument('--mask_F', type=int)
    parser.add_argument('--mask_mf', type=int)
    parser.add_argument('--mask_T', type=int)
    parser.add_argument('--mask_mt', type=int)
    parser.add_argument('--mask_ratio_t', type=float)
    parser.add_argument('--log_level', type=int, default=20, help='Log level')
    args = parser.parse_args()
    return args


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
    deepspeech = DeepSpeech.construct(config_path=config_path, alphabet_path=alphabet_path)
    if args.pretrained_weights:
        deepspeech.load(args.pretrained_weights)

    freeze(deepspeech.model)
    gpus = get_available_gpus()
    config = Configuration(config_path)
    extended_model = create_extended_model(deepspeech.model, config, is_gpu=len(gpus) > 0)

    optimizer = DeepSpeech.get_optimizer(**config.optimizer)
    loss = DeepSpeech.get_loss()
    gpus = get_available_gpus()
    deepspeech.compiled_model = DeepSpeech.compile_model(extended_model, optimizer, loss, gpus)

    train_generator = deepspeech.create_generator(args.train, batch_size=args.batch_size, source=args.source,
                                                  shuffle_after_epoch=args.shuffle_after_epoch, mask=args.mask,
                                                  mask_params=dict(F=args.mask_F, mf=args.mask_mf, T=args.mask_T,
                                                                   mt=args.mask_mt, ratio_t=args.mask_ratio_t))
    dev_generator = deepspeech.create_generator(args.dev, batch_size=args.batch_size, source=args.source)

    deepspeech.fit(train_generator, dev_generator, epochs=args.epochs, shuffle=False)
    deepspeech.save(weights_path)


if __name__ == "__main__":
    chdir(to='ROOT')
    arguments = parse_arguments()
    log_path = os.path.join(arguments.model_dir, 'training.log')
    config_path = os.path.join(arguments.model_dir, 'configuration.yaml')
    alphabet_path = os.path.join(arguments.model_dir, 'alphabet.txt')
    weights_path = os.path.join(arguments.model_dir, 'weights.hdf5')
    logger = create_logger(log_path, level=arguments.log_level, name='deepspeech')
    logger.info(f'Arguments: \n{arguments}')
    main(arguments)
