import dill
import os
import logging


def save(data, file_name):
    """ Save arbitrary data in the file. """
    with open(file_name, mode='wb') as file:
        dill.dump(data, file)


def chdir(to='ROOT'):
    """ Change the current work directory. Easily it can be switched to the project ROOT."""
    if to == 'ROOT':
        import run
        new_cwd = os.path.dirname(run.__file__)
    else:
        new_cwd = to
    os.chdir(new_cwd)


def create_logger(file_path, level=20, name='deepspeech'):
    """ Create the logger with default"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if file_path:
        handler = logging.FileHandler(file_path, mode='w')
    else:
        handler = logging.StreamHandler()

    formater = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formater)
    # handle all messages from logger (not set handler level)
    logger.addHandler(handler)
    return logger
