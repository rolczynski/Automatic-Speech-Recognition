import os
import pickle
import numpy as np
from collections import namedtuple


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


def load_model(name):
    with open(name, mode='rb') as f:
        deepspeech = pickle.load(f)
    return deepspeech


def wer(r, h):
    """ Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Source: https://martin-thoma.com/word-error-rate-calculation/ """
    import numpy as np
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]


def cache_features_to_files(name, dir, cache_files_count):
    """ """
    cache_dir = os.path.join(dir, '.cache', name)
    cache_index = os.path.join(cache_dir, '.index')

    def create_dir_and_cache_files():
        os.makedirs(cache_dir)
        with open(cache_index, 'wb') as f:
            pickle.dump({'hits': 0, 'misses': 0, 'name': name, 
                         'cache_files_count': cache_files_count}, f)

        for file_index in range(cache_files_count):
            file_name = os.path.join(cache_dir, f'cache-{file_index}')
            with open(file_name, 'wb') as f:
                pickle.dump({}, f)

    if not os.path.isdir(cache_dir):
        create_dir_and_cache_files()

    def decorating_function(user_function):
        return __cache_features_to_files(user_function, cache_dir, cache_index, cache_files_count)

    return decorating_function


def __cache_features_to_files(user_function, cache_dir, cache_index, cache_files_count):
    """ Simple caching without ordering """
    def get_cache_index():
        with open(cache_index, 'rb') as f:
            return pickle.load(f)

    def get_result(file_name, key):
        with open(file_name, 'rb') as f:
            results = pickle.load(f)
            return results[key]

    def set_result(key, result):
        file_index, = np.random.choice(cache_files_count, 1)
        file_name = os.path.join(cache_dir, f'cache-{file_index}')

        with open(cache_index, 'r+b') as f_cache:
            cache = pickle.load(f_cache)
            cache[key] = file_name

            # Write the linked file with cached results
            with open(file_name, 'r+b') as f_results:
                results = pickle.load(f_results)
                results[key] = result
                f_results.seek(0)
                pickle.dump(results, f_results)

            f_cache.seek(0)
            pickle.dump(cache, f_cache)

    def set_statistics(reset=False):
        nonlocal hits, misses
        with open(cache_index, 'r+b') as f_cache:
            results = pickle.load(f_cache)
            if reset:
                results['hits'] = 0
                results['misses'] = 0
            else:
                results['hits'] += hits
                results['misses'] += misses
                hits = misses = 0
            f_cache.seek(0)
            pickle.dump(results, f_cache)

    def cache_info():
        """Report cache statistics"""
        cache = get_cache_index()
        return _CacheInfo(cache['name'], cache['cache_files_count'], cache['hits'], cache['misses'])


    _CacheInfo = namedtuple('CacheInfo', ['name', 'cache_files_count', 'hits', 'misses'])
    i = hits = misses = 0
    set_statistics(reset=True)

    def wrapper(key):
        """ The wrapped method must have one argument (the key). """
        nonlocal i, hits, misses
        i += 1
        cache = get_cache_index()
        if key in cache:
            hits += 1
            file_name = cache[key]
            result = get_result(file_name, key)
        else:
            misses += 1
            result = user_function(key)
            set_result(key, result)

        if i % 10 == 0:
            set_statistics()

        return result

    wrapper.cache_info = cache_info
    return wrapper


def convert_to_CPU(model):
    """ This methods changes model definition to make it avialable for users
    without GPU"""
    return NotImplementedError
