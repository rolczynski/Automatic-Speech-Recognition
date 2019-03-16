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
from source.deepspeech import DeepSpeech
from source.metric import Metric, get_metrics
from source.utils import chdir, load, create_logger


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
    save_in(store, [X, *activations, y], metrics, references)
    return metrics


def evaluate(deepspeech: DeepSpeech, generator: Iterable, save_activations: bool, store_path: str) -> pd.DataFrame:
    references = pd.DataFrame(columns=['sample_id', 'transcript', 'prediction', 'wer', 'cer']).set_index('sample_id')
    get_activations = get_activations_function(deepspeech.model)
    with h5py.File(store_path, mode='w') as store:
        batch_metrics = [evaluate_batch(deepspeech, X, y, store, references, save_activations, get_activations)
                         for X, y in tqdm(generator)]
    with pd.HDFStore(store_path, mode='r+') as store:
        store.put('references', references)
    metrics = pd.DataFrame(reduce(operator.concat, batch_metrics))
    return metrics


def main(store_path: str, model_dir: str, features_store_path: str, batch_size: int, save_activations: bool):
    """ Evaluate model using prepared features. """
    deepspeech = load(model_dir)
    generator = deepspeech.create_generator(features_store_path, source='from_prepared_features', batch_size=batch_size)

    units = calculate_units(deepspeech.model)
    logger.info(f'Model contains: {units//1e6:.0f}M units ({units})')

    metrics = evaluate(deepspeech, generator, save_activations, store_path)
    logger.info(f'Mean CER: {metrics.cer.mean():.4f}')
    logger.info(f'Mean WER: {metrics.wer.mean():.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', required=True, help='File hdf5 keeps evaluation results')
    parser.add_argument('--model_dir', required=True, help='Pretrained model directory')
    parser.add_argument('--features_store', required=True, help='HDF5 Store with precomputed features')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size (depends on model) should be the same as during training')
    parser.add_argument('--log_file', help='Log file')
    parser.add_argument('--log_level', type=int, default=20, help='Log level')
    parser.add_argument('--save_activations', type=bool, default=True, help='Save all activation through evaluation')
    args = parser.parse_args()
    chdir(to='ROOT')

    logger = create_logger(args.log_file, level=args.log_level, name='evaluate')
    logger.info(f'Arguments: \n{args}')
    main(args.store, args.model_dir, args.features_store, args.batch_size, args.save_activations)
