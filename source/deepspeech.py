import os
import pickle
import tensorflow as tf
from math import exp

from keras.layers import Input
from keras.utils import multi_gpu_model
from keras.callbacks import TerminateOnNaN, LearningRateScheduler
from keras.optimizers import SGD, Adam
from keras.backend.tensorflow_backend import _get_available_gpus as get_available_gpus

# Internal source code
from source import audio, model, ctc_decoder
from source.callbacks import CustomModelCheckpoint, CustomTensorBoard, CustomEarlyStopping
from source.generator import DataGenerator
from source.text import Alphabet
from source.utils import make_keras_picklable
make_keras_picklable()


class DeepSpeech:
    """

    The public attributes after the training:
        .model                # trained Keras model
        .alphabet             # describe valid chars (Mozilla DeepSpeech format)
        .configuration        # parameters used during training
        .language_model       # support decoding (optional)

    """

    def __init__(self, configuration):
        """ Setup configuration and compile the model """
        self.configuration = configuration
        self.__set_model()
        self.__set_alphabet()


    def __call__(self, files, full=False):
        """ Easy interaction with the pre-trained model """
        X = audio.get_features_mfcc(files)
        y_hat = self.model.predict_on_batch(X)
        sentences = self.decode(y_hat)
        return X, y_hat, sentences if full else sentences


    def train(self):
        """ Train model using train and dev data generators """
        config_dataset = self.configuration.dataset
        train_generator = DataGenerator(name='train',
                                        csv_path=config_dataset.train_csv_path,
                                        alphabet=self.alphabet,
                                        **config_dataset.parameters)
        dev_generator = DataGenerator(name='dev',
                                      csv_path=config_dataset.dev_csv_path,
                                      alphabet=self.alphabet,
                                      **config_dataset.parameters)

        gpus = get_available_gpus()
        if len(gpus) > 1:
            distributed_model = multi_gpu_model(self.model, len(gpus))
        else:
            distributed_model = self.model

        optimizer = self.__get_optimizer()
        objective = self.__get_objective()
        y = Input(name='y', shape=[None], dtype='int32')
        distributed_model.compile(optimizer=optimizer,
                                  loss=objective,
                                  target_tensors=[y])

        # The template model shares the same weights, but it is not distributed
        # along different devices.
        distributed_model.template_model = self.model

        callbacks = self.__get_callbacks()

        history = distributed_model.fit_generator(
            generator=train_generator,
            validation_data=dev_generator,
            callbacks=callbacks,
            shuffle=False,
            **self.configuration.fit_generator.parameters
        )
        self.__set_best_weights_to_model(history)


    def decode(self, y_hat, lm=None):
        """ Decode probabilities along characters using beam search. The search
        can be supported by the language model """
        if lm is False:
            ...
        raise NotImplemented


    def save(self, file_path=None):
        """ Pickle the object DeepSpeech into one binary file. This is not
        supported by Keras due to limitation of Theano backend. Deepspeech use
        only TensorFlow backend. """
        if not file_path:
            file_path = os.path.join(self.configuration.exp_dir, 'model.bin')
        with open(file_path, mode='wb') as f:
            pickle.dump(self, f)


    def __set_model(self):
        """ Define model base on the experiment configuration. """
        config_model = self.configuration.model
        self.model = model.get_model(**config_model.parameters)
        if config_model.checkpoint.use:
            self.model.load_weights(config_model.checkpoint.file_path)


    def __set_alphabet(self):
        """ Alphabet consists all valid characters. """
        alphabet_path = self.configuration.dataset.alphabet_path
        self.alphabet = Alphabet(alphabet_path)


    def __get_optimizer(self):
        """ Define optimizer - use keras documentation `keras.optimizers` """
        config_optimizer = self.configuration.optimizer
        if config_optimizer.name == 'sgd':
            return SGD(**config_optimizer.parameters)
        elif config_optimizer.name == 'adam':
            return Adam(**config_optimizer.parameters)


    def __get_objective(self):
        """ The CTC loss using TensorFlow's `ctc_loss` using Keras backend """
        def get_length(tensor):
            lengths = tf.reduce_sum(tf.ones_like(tensor), 1)
            return tf.reshape(tf.cast(lengths, tf.int32), [-1, 1])

        def ctc_objective(y, y_hat):
            sequence_length = get_length(tf.reduce_max(y_hat, 2))
            label_length = get_length(y)
            return tf.keras.backend.ctc_batch_cost(y, y_hat, sequence_length, label_length)
        return ctc_objective


    def __get_callbacks(self):
        """ Define callbacks to get a view on internal states of the model """
        callbacks = []
        for config_callback in self.configuration.callbacks:
            if config_callback.name == 'TerminateOnNaN':
                callbacks.append(TerminateOnNaN())

            elif config_callback.name == 'CustomEarlyStopping':
                callbacks.append(CustomEarlyStopping(**config_callback.parameters))

            elif config_callback.name == 'LearningRateScheduler':
                args = config_callback.parameters
                step_decay = lambda epoch, lr: lr*exp(-epoch*args.k)
                callbacks.append(LearningRateScheduler(step_decay, args.verbose))

            elif config_callback.name == 'CustomModelCheckpoint':
                log_dir = os.path.join(self.configuration.exp_dir, 'checkpoints')
                callbacks.append(CustomModelCheckpoint(log_dir))

            elif config_callback.name == 'CustomTensorBoard':
                log_dir = os.path.join(self.configuration.exp_dir, 'tensorboard')
                callbacks.append(CustomTensorBoard(log_dir))
        return callbacks


    def __set_best_weights_to_model(self, history_callback):
        """ Set best weights to the model. Checkpoint callback save the best
        weights path. """
        file_path = history_callback.best_weights_path
        self.model.load_weights(file_path)
