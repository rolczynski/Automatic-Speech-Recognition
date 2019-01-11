import os
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback, TensorBoard, EarlyStopping


class CustomModelCheckpoint(Callback):
    """ Save model architecture and weights for the single or multi-gpu
    model. """

    def __init__(self, log_dir):
        """ Create directory where the files are stored if needed """
        super().__init__()
        self.log_dir = log_dir
        self.best_result = np.inf


    def _create_directory(self, _):
        """ Create the directory where the checkpoints are saved. """
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)

    on_train_begin = _create_directory


    def _save_model(self, epoch, logs={}):
        """ Save model with weights of the single-gpu template model. """
        val_loss = logs.get('val_loss')
        name = f'weights.{epoch + 1:02d}-{val_loss:.2f}.hdf5'
        file_path = os.path.join(self.log_dir, name)
        self.model.template_model.save_weights(file_path, overwrite=True)
        if val_loss < self.best_result:
            self.best_result = val_loss
            self.model.history.best_weights_path = file_path

    on_epoch_end = _save_model


class CustomTensorBoard(TensorBoard):
    """ This callback enable to save the batch logs. Write images and grads are
    disable. The generator is required and not supported with fit_generator. """

    def __init__(self, log_dir):
        super().__init__(log_dir)
        self.processed_batches = 0


    def _save_batch_loss(self, _, logs={}):
        """ Add value to the tensorboard event """
        loss = logs.get('loss')
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = loss
        summary_value.tag = 'Loss (each batch)'
        self.writer.add_summary(summary, self.processed_batches)
        self.writer.flush()
        self.processed_batches += 1

    on_batch_end = _save_batch_loss


class CustomEarlyStopping(EarlyStopping):
    """ The callback stops training if the minimal target is not achieved. """

    def __init__(self, **kwargs):
        mini_targets = kwargs.pop('mini_targets')
        self._mini_targets = mini_targets
        super().__init__(**kwargs)


    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        current = logs.get(self.monitor)
        if epoch in self._mini_targets and current > self._mini_targets[epoch]:
            self.model.stop_training = True
