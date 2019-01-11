import dill
import os
import logging


def make_keras_picklable():
    from tempfile import NamedTemporaryFile as temp
    from keras.engine.saving import save_model, load_model as load_keras_model
    from keras.models import Model

    def __getstate__(self):
        with temp(suffix='.hdf5', delete=True) as f:
            save_model(self, f.name, overwrite=True)
            model_str = f.read()
        return {'model_str': model_str}

    def __setstate__(self, state):
        with temp(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = load_keras_model(fd.name)
        self.__dict__ = model.__dict__

    Model.__getstate__ = __getstate__
    Model.__setstate__ = __setstate__


def load_model(model_path):
    """ Load model even without the deepspeech package. """
    with open(model_path, mode='rb') as f:
        deepspeech = dill.load(f)
    return deepspeech


def chdir(to='ROOT'):
    """ Change the current work directory. Easily it can be switched to the project ROOT."""
    if to == 'ROOT':
        import run
        new_cwd = os.path.dirname(run.__file__)
    else:
        new_cwd = to
    os.chdir(new_cwd)


def create_logger(file_path, level=10, name='deepspeech'):
    """ Create the logger with default"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.FileHandler(file_path, mode='w')  # handle all messages from logger
    formater = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formater)

    logger.addHandler(handler)
    return logger
