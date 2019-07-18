import os
from scripts.run import parse_arguments, create_generators
from scripts.run_extension import freeze, create_extended_model
from source.deepspeech import DeepSpeech, Configuration, get_available_gpus
from source.utils import chdir, create_logger


def load_extended_model(config_path, alphabet_path, weights_path):
    deepspeech = DeepSpeech.construct(config_path=config_path, alphabet_path=alphabet_path)

    freeze(deepspeech.model)
    gpus = get_available_gpus()
    config = Configuration(config_path)
    extended_model = create_extended_model(deepspeech.model, config, is_gpu=len(gpus) > 0)

    optimizer = DeepSpeech.get_optimizer(**config.optimizer)
    loss = DeepSpeech.get_loss()
    gpus = get_available_gpus()
    deepspeech.model = extended_model
    deepspeech.compiled_model = DeepSpeech.compile_model(extended_model, optimizer, loss, gpus)
    deepspeech.load(weights_path)
    return deepspeech


def main(args):
    deepspeech = load_extended_model(CONFIG_PATH, ALPHABET_PATH, WEIGHTS_PATH)
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
