import argparse
import os
import sys  # Add `source` module (needed when it runs via terminal)
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import pandas as pd
from collections import namedtuple
from functools import reduce

from source.utils import chdir, load_model, create_logger
from source.generator import DataGenerator
from source.text import get_batch_transcripts
from scripts.wer import edit_distance


def calculate_units(model):
    """ Calculate number of the model parameters. """
    units = 0
    for parameters in model.get_weights():
        units += reduce(lambda x, y: x * y, parameters.shape)
    return units


def get_results(sources, destinations):
    """ Calculate base metrics: WER and CER. """
    Sample = namedtuple('Sample', 'original prediction wer cer')
    for source, destination in zip(sources, destinations):
        wer_distance, *_ = edit_distance(source.split(), destination.split())
        wer = wer_distance / len(destination.split())

        cer_distance, *_ = edit_distance(source, destination)
        cer = cer_distance / len(destination)

        yield Sample(destination, source, wer, cer)


def main(model_path, store_path, batch_size, home_directory):
    """ Evaluate model using prepared features. """
    deepspeech = load_model(model_path)
    generator = DataGenerator.from_prepared_features(file_path=store_path,
                                                     alphabet=deepspeech.alphabet,
                                                     batch_size=batch_size)
    units = calculate_units(deepspeech.model)
    logger.info(f'Model contains: {units//1e6:.0f}M units ({units})')

    results = []
    for index, (X, y) in enumerate(generator):
        logger.info(f'Batch ({index})')
        y_hat = deepspeech.predict_on_batch(X)

        predict_sentences = deepspeech.decode(y_hat, beam_size=1000)
        true_sentences = get_batch_transcripts(y, deepspeech.alphabet)

        batch_results = get_results(sources=predict_sentences, destinations=true_sentences)
        results.extend(list(batch_results))

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(home_directory, 'results.csv'), index=False)
    logger.info(f'Mean CER: {df.cer.mean():.4f}')
    logger.info(f'Mean WER: {df.wer.mean():.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Pretrained model')
    parser.add_argument('--home_directory', required=True, help='Directory where save results')
    parser.add_argument('--store', required=True, help='Store with precomputed features')
    parser.add_argument('--log_file', help='Log file')
    parser.add_argument('--log_level', type=int, default=20, help='Log level')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    arguments = parser.parse_args()
    chdir(to='ROOT')

    os.makedirs(arguments.home_directory, exist_ok=True)
    logger = create_logger(arguments.log_file, level=arguments.log_level, name='evaluate')
    logger.info(f'Arguments: \n{arguments}')
    main(arguments.model, arguments.store, arguments.batch_size, arguments.home_directory)
