import os
import argparse
from typing import Tuple

from source.generator import DataGenerator, DistributedDataGenerator
from source.deepspeech import DeepSpeech
from source.utils import chdir, create_logger


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help='DeepSpeech model directory')
    parser.add_argument('--train', required=True, help='Train source (csv/hdf5 file)')
    parser.add_argument('--dev', required=True, help='Dev source (csv/hdf5 file)')
    parser.add_argument('--source', choices=['from_audio_files', 'from_prepared_features'], required=True)
    parser.add_argument('--source_distributed', dest='source_distributed', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (int)')
    parser.add_argument('--epochs', type=int, default=25, help='Epochs during training')
    parser.add_argument('--shuffle_after_epoch', type=int, default=1, help='Shuffle generator indices after epoch')
    parser.add_argument('--pretrained_weights', help='Use weights from the pretrained model')
    parser.add_argument('--mask', dest='mask', action='store_true', help='Mask features during training')
    parser.add_argument('--mask_F', type=int)
    parser.add_argument('--mask_mf', type=int)
    parser.add_argument('--mask_Tmin', type=int)
    parser.add_argument('--mask_Tmax', type=int)
    parser.add_argument('--mask_mt', type=int)
    parser.add_argument('--mask_ratio_t', type=float)
    parser.add_argument('--log_level', type=int, default=20, help='Log level')
    return parser


def create_generators(deepspeech: DeepSpeech, args) -> Tuple[DataGenerator, DataGenerator]:
    train_gen_params = dict(
        alphabet=deepspeech.alphabet,
        features_extractor=deepspeech.features_extractor,
        batch_size=args.batch_size,
        shuffle_after_epoch=args.shuffle_after_epoch,
        mask=args.mask,
        mask_params=dict(F=args.mask_F, mf=args.mask_mf, Tmin=args.mask_Tmin,
                         Tmax=args.mask_Tmax, mt=args.mask_mt, ratio_t=args.mask_ratio_t)
    )
    if args.source_distributed:
        generator_constructor = getattr(DistributedDataGenerator, args.source)
        train_gen_params['file_paths'] = args.train.split(',')
    else:
        generator_constructor = getattr(DataGenerator, args.source)
        train_gen_params['file_path'] = args.train

    train_generator = generator_constructor(**train_gen_params)
    dev_generator = getattr(DataGenerator, args.source)(        # Dev generator can not be distributed
        args.dev,
        alphabet=deepspeech.alphabet,
        features_extractor=deepspeech.features_extractor,
        batch_size=args.batch_size
    )
    return train_generator, dev_generator


def setup_deepspeech(config_path: str, alphabet_path: str, pretrained_weights: str = '') -> DeepSpeech:
    deepspeech = DeepSpeech.construct(config_path, alphabet_path)
    if pretrained_weights:
        deepspeech.load(pretrained_weights)
    return deepspeech


def main(args):
    deepspeech = setup_deepspeech(CONFIG_PATH, ALPHABET_PATH, pretrained_weights=args.pretrained_weights)
    train_generator, dev_generator = create_generators(deepspeech, args)
    deepspeech.fit(train_generator, dev_generator, epochs=args.epochs, shuffle=False)
    deepspeech.save(WEIGHTS_PATH)


if __name__ == "__main__":
    chdir(to='ROOT')
    parser = create_parser()

    ARGUMENTS = parser.parse_args()
    LOG_PATH = os.path.join(ARGUMENTS.model_dir, 'training.log')
    CONFIG_PATH = os.path.join(ARGUMENTS.model_dir, 'configuration.yaml')
    ALPHABET_PATH = os.path.join(ARGUMENTS.model_dir, 'alphabet.txt')
    WEIGHTS_PATH = os.path.join(ARGUMENTS.model_dir, 'weights.hdf5')

    logger = create_logger(LOG_PATH, level=ARGUMENTS.log_level, name='deepspeech')
    logger.info(f'Arguments: \n{ARGUMENTS}')

    main(ARGUMENTS)
