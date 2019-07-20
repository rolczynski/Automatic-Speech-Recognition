import os
# Activation can not be handled using distrubuted model (few GPUs). First GPU selected.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from source.generator import DataGenerator
from source.utils import chdir, create_logger
from scripts.evaluate import parse_arguments, calculate_units, evaluate
from scripts.run_extension_tune import load_extended_model


def main(store_path: str, features_store_path: str, batch_size: int, save_activations: bool,
         mask: bool, mask_F: int, mask_mf: int, mask_T: int, mask_mt: int, mask_ratio_t: float):
    """ Evaluate model using prepared features. """
    deepspeech = load_extended_model(CONFIG_PATH, ALPHABET_PATH, WEIGHTS_PATH)
    generator = DataGenerator.from_prepared_features(
        features_store_path,
        alphabet=deepspeech.alphabet,
        features_extractor=deepspeech.features_extractor,
        batch_size=batch_size,
        mask=mask,
        mask_params=dict(F=mask_F, mf=mask_mf, T=mask_T,
                         mt=mask_mt, ratio_t=mask_ratio_t)
    )
    units = calculate_units(deepspeech.model)
    logger.info(f'Model contains: {units//1e6:.0f}M units ({units})')

    metrics = evaluate(deepspeech, generator, save_activations, store_path)
    logger.info(f'Mean CER: {metrics.cer.mean():.4f}')
    logger.info(f'Mean WER: {metrics.wer.mean():.4f}')


if __name__ == '__main__':
    ARGUMENTS = parse_arguments()
    chdir(to='ROOT')

    CONFIG_PATH = os.path.join(ARGUMENTS.model_dir, 'configuration.yaml')
    ALPHABET_PATH = os.path.join(ARGUMENTS.model_dir, 'alphabet.txt')
    WEIGHTS_PATH = os.path.join(ARGUMENTS.model_dir, 'weights.hdf5')

    logger = create_logger(ARGUMENTS.log_file, level=ARGUMENTS.log_level, name='evaluate')
    logger.info(f'Arguments: \n{ARGUMENTS}')

    main(
        store_path=ARGUMENTS.store,
        features_store_path=ARGUMENTS.features_store,
        batch_size=ARGUMENTS.batch_size,
        save_activations=ARGUMENTS.save_activations,
        mask=ARGUMENTS.mask,
        mask_F=ARGUMENTS.mask_F,
        mask_mf=ARGUMENTS.mask_mf,
        mask_T=ARGUMENTS.mask_T,
        mask_mt=ARGUMENTS.mask_mt,
        mask_ratio_t=ARGUMENTS.mask_ratio_t
    )
