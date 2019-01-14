import os
import argparse
from source.deepspeech import DeepSpeech
from source.generator import DataGenerator
from source.utils import chdir, create_logger


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration', required=True, help='DeepSpeech configuration file')
    parser.add_argument('--home_directory', required=True, help='Experiment home directory')
    parser.add_argument('--train', required=True, help='Train source (csv/hdf5 file)')
    parser.add_argument('--dev', required=True, help='Dev source (csv/hdf5 file)')
    parser.add_argument('--source', choices=['from_audio_files', 'from_prepared_features'], required=True)
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (int)')
    parser.add_argument('--epochs', type=int, default=25, help='Epochs during training')
    parser.add_argument('--shuffle_after_epoch', type=int, default=1, help='Shuffle generator indices after epoch')
    parser.add_argument('--pretrained', help='Fit the pretrained model')
    parser.add_argument('--log_file', help='Log file')
    parser.add_argument('--log_level', type=int, default=20, help='Log level')
    args = parser.parse_args()
    return args


def fit(args):
    # Set up DeepSpeech object (and load pretrained model if provided)
    deepspeech = DeepSpeech.from_configuration(file_path=args.configuration)
    if args.pretrained:
        deepspeech.copy_weights(args.pretrained)

    # Create generator from prepared features or process raw audio files (audio paths saved in csv)
    create_generator = getattr(DataGenerator, args.source)
    train_generator = create_generator(file_path=args.train,
                                       alphabet=deepspeech.alphabet,
                                       batch_size=args.batch_size,
                                       shuffle_after_epoch=args.shuffle_after_epoch)

    dev_generator = create_generator(file_path=args.dev,
                                     alphabet=deepspeech.alphabet,
                                     batch_size=args.batch_size)

    chdir(to=args.home_directory)  # The paths can be defined as relative (e.g. used in the tensorboard)

    # Choose `model.fit_generator` parameters. This allows you to do features
    # extraction on CPU in parallel to training your model on GPU's.
    deepspeech.fit_generator(train_generator, dev_generator, epochs=args.epochs, shuffle=False)
    deepspeech.save('model.bin')

    # Model can be loaded even without Deepspeech-Keras installation (only dependencies required)
    # from deepspeech.utils import load_model
    # deepspeech = load_model('model.bin')


if __name__ == "__main__":
    arguments = parse_arguments()
    os.makedirs(arguments.home_directory, exist_ok=True)
    logger = create_logger(arguments.log_file, level=arguments.log_level, name='deepspeech')
    logger.info(f'Arguments: \n{arguments}')
    fit(arguments)
