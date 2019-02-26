import os
import argparse
from source.deepspeech import DeepSpeech
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
    deepspeech = DeepSpeech.from_configuration(file_path=args.configuration)
    if args.pretrained:
        deepspeech.load(path=args.pretrained)

    train_generator = deepspeech.create_generator(args.train, batch_size=args.batch_size, source=args.source,
                                                  shuffle_after_epoch=args.shuffle_after_epoch)
    dev_generator = deepspeech.create_generator(args.dev, batch_size=args.batch_size, source=args.source)

    chdir(to=args.home_directory)     # The paths can be defined as relative (e.g. used in the tensorboard)
    deepspeech.fit(train_generator, dev_generator, epochs=args.epochs, shuffle=False)
    deepspeech.save('weights.hdf5')


if __name__ == "__main__":
    arguments = parse_arguments()
    os.makedirs(arguments.home_directory, exist_ok=True)
    logger = create_logger(arguments.log_file, level=arguments.log_level, name='deepspeech')
    logger.info(f'Arguments: \n{arguments}')
    fit(arguments)
