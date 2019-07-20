import os
# Activation can not be handled using distrubuted model (few GPUs). First GPU selected.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import operator
from functools import reduce
from typing import List, Callable, Iterable
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras import backend as K
from keras.models import Model
from source.generator import DataGenerator
from source.deepspeech import DeepSpeech
from source.metric import Metric, get_metrics
from source.utils import chdir, load, create_logger


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', required=True, help='File hdf5 keeps evaluation results')
    parser.add_argument('--model_dir', required=True, help='Pretrained model directory')
    parser.add_argument('--features_store', required=True, help='HDF5 Store with precomputed features')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size (depends on model) should be the same as during training')
    parser.add_argument('--log_file', help='Log file')
    parser.add_argument('--log_level', type=int, default=20, help='Log level')
    parser.add_argument('--save_activations', dest='save_activations', action='store_true', help='Save all activation through evaluation')
    parser.add_argument('--mask', dest='mask', action='store_true', help='Mask features during training')
    parser.add_argument('--mask_F', type=int)
    parser.add_argument('--mask_mf', type=int)
    parser.add_argument('--mask_T', type=int)
    parser.add_argument('--mask_mt', type=int)
    parser.add_argument('--mask_ratio_t', type=float)
    args = parser.parse_args()
    return args


def calculate_units(model: Model) -> int:
    """ Calculate number of the model parameters. """
    units = 0
    for parameters in model.get_weights():
        units += reduce(lambda x, y: x * y, parameters.shape)
    return units


def get_activations_function(model: Model) -> Callable:
    """ Function which handle all activations through one pass. """
    inputs = [model.input, K.learning_phase()]
    outputs = [layer.output for layer in model.layers][1:]
    return K.function(inputs, outputs)


def save_in(store: h5py.File, layer_outputs: List[np.ndarray], metrics: List[Metric], references: pd.DataFrame):
    """ Save batch data into HDF5 file. """
    for index, metric in enumerate(metrics):
        sample_id = len(references)
        references.loc[sample_id] = metric

        for output_index, batch_layer_outputs in enumerate(layer_outputs):
            layer_output = batch_layer_outputs[index]
            store.create_dataset(f'outputs/{output_index}/{sample_id}', data=layer_output)


def evaluate_batch(deepspeech: DeepSpeech, X: np.ndarray, y: np.ndarray, store: h5py.File,
                   references: pd.DataFrame, save_activations: bool, get_activations: Callable) -> List[Metric]:
    if save_activations:
        *activations, y_hat = get_activations([X, 0])  # Learning phase is `test=0`
    else:
        activations = []
        y_hat = deepspeech.predict(X)

    predict_sentences = deepspeech.decode(y_hat)
    true_sentences = deepspeech.get_transcripts(y)
    metrics = list(get_metrics(sources=predict_sentences, destinations=true_sentences))
    save_in(store, [X, *activations, y_hat], metrics, references)
    return metrics


def evaluate(deepspeech: DeepSpeech, generator: Iterable, save_activations: bool, store_path: str) -> pd.DataFrame:
    references = pd.DataFrame(columns=['sample_id', 'transcript', 'prediction', 'wer', 'cer']).set_index('sample_id')
    get_activations = get_activations_function(deepspeech.model) if save_activations else None

    with h5py.File(store_path, mode='w') as store:
        batch_metrics = [evaluate_batch(deepspeech, X, y, store, references, save_activations, get_activations)
                         for X, y in tqdm(generator)]
    with pd.HDFStore(store_path, mode='r+') as store:
        store.put('references', references)
    metrics = pd.DataFrame(reduce(operator.concat, batch_metrics))
    return metrics


def main(store_path: str, model_dir: str, features_store_path: str, batch_size: int, save_activations: bool,
         mask: bool, mask_F: int, mask_mf: int, mask_T: int, mask_mt: int, mask_ratio_t: float):
    """ Evaluate model using prepared features. """
    deepspeech = load(model_dir)
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
        model_dir=ARGUMENTS.model_dir,
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
