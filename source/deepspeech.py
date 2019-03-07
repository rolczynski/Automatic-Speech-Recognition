import os
import numpy as np
import tensorflow as tf
from math import exp
from functools import partial
from typing import List, Callable
from keras import Model
from keras.layers import Input
from keras.utils import multi_gpu_model
from keras.callbacks import Callback, TerminateOnNaN, LearningRateScheduler, History
from keras.optimizers import Optimizer, SGD, Adam
from keras.backend.tensorflow_backend import _get_available_gpus as get_available_gpus

from source import audio, model, text, ctc_decoder, configuration, utils
from source.text import Alphabet
from source.audio import FeaturesExtractor
from source.generator import DataGenerator
from source.callbacks import CustomModelCheckpoint, CustomTensorBoard, CustomEarlyStopping, ResultKeeper
from source.configuration import Configuration


class DeepSpeech:
    """
    The public attributes after the training:
        - model: Keras model
        - alphabet: describe valid chars
        - callbacks: get a view on internal states during training
    """

    def __init__(self,
                 model: Model,
                 loss: Callable,
                 optimizer: Optimizer,
                 callbacks: List[Callback],
                 alphabet: Alphabet,
                 decoder: Callable,
                 features_extractor: FeaturesExtractor,
                 gpus: list = []):
        """ Setup configuration and compile the model """
        self.model = model
        self.alphabet = alphabet
        self.features_extractor = features_extractor
        self.decoder = decoder
        self.callbacks = callbacks
        self.gpus = gpus
        self.distributed_model = self.compile_model(model, optimizer, loss, gpus)


    @classmethod
    def construct(cls, config_path: str, alphabet_path: str) -> 'DeepSpeech':
        """ Construct DeepSpeech object base on the configuration and the alphabet files. """
        config = Configuration(config_path)
        model_dir = os.path.dirname(config_path)
        gpus = get_available_gpus()

        model = cls.get_model(is_gpu=len(gpus) > 0, **config.model)
        loss = cls.get_loss()
        optimizer = cls.get_optimizer(**config.optimizer)
        callbacks = cls.get_callbacks(home_dir=model_dir, configurations=config.callbacks)

        alphabet = cls.get_alphabet(alphabet_path)
        features_extractor = cls.get_features_extractor(**config.features_extractor)
        decoder = cls.get_decoder(alphabet=alphabet, model=model, **config.decoder)
        return cls(model, loss, optimizer, callbacks, alphabet, decoder, features_extractor, gpus)


    def __call__(self, files: List[str]) -> List[str]:
        """ Easy interaction with the trained model """
        X = self.get_features(files)
        y_hat = self.predict(X)
        sentences = self.decode(y_hat)
        return sentences


    def get_features(self, files: List[str]) -> np.ndarray:
        """ Extract features from files. """
        return self.features_extractor.get_features_mfcc(files)


    def get_labels(self, transcripts: List[str]) -> np.ndarray:
        """ Convert transcripts to labels. """
        return self.alphabet.get_batch_labels(transcripts)


    def get_transcripts(self, labels: np.ndarray) -> List[str]:
        """ Convert labels to transcripts. """
        return self.alphabet.get_batch_transcripts(labels)


    def create_generator(self, file_path, source='from_audio_files', **kwargs) -> DataGenerator:
        """ Create generator from audio files (csv file) or prepared features (hdf5 file). """
        _create_generator = getattr(DataGenerator, source)
        return _create_generator(file_path, alphabet=self.alphabet, features_extractor=self.features_extractor, **kwargs)


    def fit(self, train_generator, dev_generator, **kwargs) -> History:
        """ Train model using train and dev data generators base on the Keras method."""
        return self.distributed_model.fit_generator(generator=train_generator,
                                                    validation_data=dev_generator,
                                                    callbacks=self.callbacks,
                                                    **kwargs)


    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Predict on the batch. """
        return self.distributed_model.predict_on_batch(X)


    def decode(self, y_hat: np.ndarray) -> List[str]:
        """ Decode probabilities along characters using the beam search algorithm. """
        return self.decoder(y_hat)


    def save(self, file_path: str):
        """ Save model weights. Object can be easily reinitialized. """
        self.model.save_weights(file_path)


    @utils.pretrained_models
    def load(self, file_path: str):
        """ Load model weights from the pretrained model. """
        self.model.load_weights(file_path)


    @staticmethod
    def compile_model(model: Model, optimizer: Optimizer, loss: Callable, gpus: list) -> Model:
        """ Compiled the model. The template model shares the same weights, but it is not distributed
        along different devices. It is useful for callbacks."""
        gpus_num = len(gpus)
        distributed_model = multi_gpu_model(model, gpus_num) if gpus_num > 1 else model
        y = Input(name='y', shape=[None], dtype='int32')
        distributed_model.compile(optimizer, loss, target_tensors=[y])
        distributed_model.template_model = model
        return distributed_model


    @staticmethod
    def get_configuration(file_path: str) -> Configuration:
        """ Read components parameters from the yaml file via Configuration object. """
        return configuration.Configuration(file_path)


    @staticmethod
    def get_model(name: str, **kwargs) -> Model:
        """ Define model base on the experiment configuration. """
        if name == 'deepspeech':
            return model.deepspeech(**kwargs)
        elif name == 'deepspeech-custom':
            return model.deepspeech_custom(**kwargs)
        raise ValueError('Wrong model name')


    @staticmethod
    def get_alphabet(file_path) -> Alphabet:
        """ Alphabet consists all valid characters / phonemes and helps work with texts. """
        return text.Alphabet(file_path)


    @staticmethod
    def get_features_extractor(**kwargs) -> FeaturesExtractor:
        """ Feature Extractor helps to convert audio files to features. """
        return audio.FeaturesExtractor(**kwargs)


    @staticmethod
    def get_optimizer(name: str, **kwargs) -> Optimizer:
        """ Define optimizer - use keras documentation `keras.optimizers`. """
        if name == 'sgd':
            return SGD(**kwargs)
        elif name == 'adam':
            return Adam(**kwargs)
        raise ValueError('Wrong optimizer name')


    @staticmethod
    def get_loss() -> Callable:
        """ The CTC loss using TensorFlow's `ctc_loss` using Keras backend. """
        def get_length(tensor):
            lengths = tf.reduce_sum(tf.ones_like(tensor), 1)
            return tf.reshape(tf.cast(lengths, tf.int32), [-1, 1])

        def ctc_loss(y, y_hat):
            sequence_length = get_length(tf.reduce_max(y_hat, 2))
            label_length = get_length(y)
            return tf.keras.backend.ctc_batch_cost(y, y_hat, sequence_length, label_length)
        return ctc_loss


    @staticmethod
    def get_decoder(name: str, alphabet: Alphabet, model: Model, **kwargs) -> Callable:
        """ Additionally can be added the warp-ctc (GPU support). """
        if name == 'naive':
            return partial(ctc_decoder.batch_naive_decode, alphabet=alphabet, **kwargs)
        elif name == 'tensorflow':
            decoder = ctc_decoder.get_tensorflow_decoder(model.output, **kwargs)
            return partial(ctc_decoder.batch_tensorflow_decode, alphabet=alphabet, decoder=decoder)
        raise ValueError('Wrong decoder name')


    @staticmethod
    def get_callbacks(home_dir: str, configurations: list) -> List[Callback]:
        """ Define callbacks to get a view on internal states during training. """
        callbacks = []
        for configuration in configurations:
            name = configuration.pop('name')

            if name == 'TerminateOnNaN':
                callbacks.append(TerminateOnNaN())

            elif name == 'ResultKeeper':
                file_path = os.path.join(home_dir, configuration.pop('file_name'))
                callbacks.append(ResultKeeper(file_path))

            elif name == 'CustomEarlyStopping':
                callbacks.append(CustomEarlyStopping(**configuration))

            elif name == 'LearningRateScheduler':
                k = configuration.pop('k')
                lr_decay = lambda epoch, lr: lr/np.power(k, epoch)
                callbacks.append(LearningRateScheduler(lr_decay, **configuration))

            elif name == 'CustomModelCheckpoint':
                log_dir = os.path.join(home_dir, configuration.pop('dir_name'))
                callbacks.append(CustomModelCheckpoint(log_dir))

            elif name == 'CustomTensorBoard':
                log_dir = os.path.join(home_dir, configuration.pop('dir_name'))
                callbacks.append(CustomTensorBoard(log_dir))
        return callbacks
