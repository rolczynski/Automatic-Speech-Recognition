import os
import dill
import numpy as np
import tensorflow as tf
from math import exp
from keras.layers import Input
from keras.utils import multi_gpu_model
from keras.callbacks import TerminateOnNaN, LearningRateScheduler
from keras.optimizers import SGD, Adam
from keras.backend.tensorflow_backend import _get_available_gpus as get_available_gpus

from source import audio, model, text, ctc_decoder, utils
from source.callbacks import CustomModelCheckpoint, CustomTensorBoard, CustomEarlyStopping, ResultKeeper
from source.configuration import Configuration
from source.utils import make_keras_picklable
make_keras_picklable()


class DeepSpeech:
    """
    The public attributes after the training:
        - model: Keras model
        - alphabet: describe valid chars
        - callbacks: get a view on internal states during training
    """

    def __init__(self,
                 model_params: dict,
                 alphabet_params: dict,
                 optimizer_params: dict,
                 callbacks_params: list):
        """ Setup configuration and compile the model """
        self._model_params = model_params
        self._alphabet_params = alphabet_params
        self._optimizer_params = optimizer_params
        self._callbacks_params = callbacks_params
        self._gpus = get_available_gpus()
        self.is_gpu = len(self._gpus) > 0   # If GPUs available, use them by default
        self.model = self._get_model(is_gpu=self.is_gpu, **model_params)
        self.alphabet = self._get_alphabet(**alphabet_params)


    @classmethod
    def from_configuration(cls, file_path: str):
        """ Create empty DeepSpeech object base on the configuration file. """
        config = Configuration(file_path)
        return cls(config.model, config.alphabet, config.optimizer, config.callbacks)


    def fit_generator(self, train_generator, dev_generator, **kwargs):
        """ Train model using train and dev data generators base on the Keras method."""
        gpus_num = len(self._gpus)
        if gpus_num > 1:
            distributed_model = multi_gpu_model(self.model, gpus_num)
        else:
            distributed_model = self.model

        y = Input(name='y', shape=[None], dtype='int32')
        # Due to problem with the optimizer/objective serialization
        # the initialization has to be placed here
        optimizer = self._get_optimizer(**self._optimizer_params)
        objective = self._get_objective()
        distributed_model.compile(optimizer=optimizer,
                                  loss=objective,
                                  target_tensors=[y])

        # The template model shares the same weights, but it is not distributed
        # along different devices. It is useful for callbacks.
        distributed_model.template_model = self.model

        callbacks = self._get_callbacks(self._callbacks_params)
        history = distributed_model.fit_generator(
            generator=train_generator,
            validation_data=dev_generator,
            callbacks=callbacks,
            **kwargs
        )
        self._set_best_weights_to_model(history)
        return callbacks


    def __call__(self, files: list):
        """ Easy interaction with the trained model """
        X = audio.get_features_mfcc(files)
        y_hat = self.model.predict_on_batch(X)
        sentences = self.decode(y_hat)
        return sentences


    def predict_on_batch(self, X):
        """ Predict on the batch. """
        return self.model.predict_on_batch(X)


    def decode(self, y_hat: np.ndarray, beam_size=100, prune=0.001, naive=False):
        """ Decode probabilities along characters using the beam search algorithm.
        Additionally can be added the warp-ctc (GPU support). """
        if not naive:
            output_tensor = self.model.output
            decode = ctc_decoder.get_tf_decoder(output_tensor, beam_size)
            labels, = decode([y_hat])
            return text.get_batch_transcripts(labels, self.alphabet)
        else:
            return ctc_decoder.batch_naive_decode(y_hat, self.alphabet, beam_size=beam_size, prune=prune)


    def copy_weights(self, model_path: str):
        """ Copy weights from the pretrained model. """
        pretrained_model = utils.load_model(model_path)
        weights = pretrained_model.model.get_weights()
        self.model.set_weights(weights)


    def save(self, model_path: str):
        """ Pickle the object DeepSpeech into one binary file. This is not
        supported by Keras due to limitation of the Theano backend. """
        with open(model_path, mode='wb') as file:
            self.to_cpu()   # Always save the CPU model (then everyone can use is)
            dill.dump(self, file)


    def to_cpu(self):
        """ Convert the model which supports the CPU. """
        cpu_model = self._get_model(is_gpu=False, **self._model_params)
        self.model.save_weights('temp')     # Use temp file because problems occurs
        cpu_model.load_weights('temp')      # with loading weights directly from CuDNNLSTM
        os.remove('temp')                   # (also precision does not match: float/double)
        self.model = cpu_model


    def _get_model(self, **kwargs):
        """ Define model base on the experiment configuration. """
        return model.get_model(**kwargs)


    def _get_alphabet(self, **kwargs):
        """ Alphabet consists all valid characters / phonemes. """
        return text.Alphabet(**kwargs)


    def _get_optimizer(self, name: str, **kwargs):
        """ Define optimizer - use keras documentation `keras.optimizers`. """
        if name == 'sgd':
            return SGD(**kwargs)
        elif name == 'adam':
            return Adam(**kwargs)


    def _get_objective(self):
        """ The CTC loss using TensorFlow's `ctc_loss` using Keras backend. """
        def get_length(tensor):
            lengths = tf.reduce_sum(tf.ones_like(tensor), 1)
            return tf.reshape(tf.cast(lengths, tf.int32), [-1, 1])

        def ctc_objective(y, y_hat):
            sequence_length = get_length(tf.reduce_max(y_hat, 2))
            label_length = get_length(y)
            return tf.keras.backend.ctc_batch_cost(y, y_hat, sequence_length, label_length)
        return ctc_objective


    def _get_callbacks(self, configurations: list):
        """ Define callbacks to get a view on internal states during training. """
        callbacks = []
        for configuration in configurations:
            name = configuration.pop('name')

            if name == 'TerminateOnNaN':
                callbacks.append(TerminateOnNaN())

            elif name == 'ResultKeeper':
                callbacks.append(ResultKeeper(**configuration))

            elif name == 'CustomEarlyStopping':
                callbacks.append(CustomEarlyStopping(**configuration))

            elif name == 'LearningRateScheduler':
                k = configuration.pop('k')
                step_decay = lambda epoch, lr: lr*exp(-epoch*k)
                callbacks.append(LearningRateScheduler(step_decay, **configuration))

            elif name == 'CustomModelCheckpoint':
                callbacks.append(CustomModelCheckpoint(**configuration))

            elif name == 'CustomTensorBoard':
                callbacks.append(CustomTensorBoard(**configuration))
        return callbacks


    def _set_best_weights_to_model(self, history):
        """ Set best weights to the model. Checkpoint callback save the best
        weights path. """
        if hasattr(history, 'best_weights_path'):
            file_path = history.best_weights_path
            self.model.load_weights(file_path)
        else:
            raise Warning('DeepSpeech can not set the best weights. CustomModelCheckpoint is required')
