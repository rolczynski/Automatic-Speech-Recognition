import numpy as np
import pandas as pd
import itertools
from typing import Tuple, List
from collections import defaultdict, Counter
import argparse
import os
import sys  # Add `source` module (needed when it runs via terminal)
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from source.utils import chdir, create_logger
from source.text import Alphabet
from scripts.plots import save_confusion_matrix, save_donut


def edit_distance(source: List[str], destination: List[str]) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Calculation of edit distance between two sequences.

    Pose the question:
    - how many steps are required to transform sequence `source` to `destination`

    This means Levenshtein distance with the substitution cost equals 1.
    This is the iterative method with the full matrix support.
    O(nm) time and space complexity.

    References:
        - https://web.stanford.edu/class/cs124/lec/med.pdf
        - https://www.python-course.eu/levenshtein_distance.php
        - https://stackabuse.com/levenshtein-distance-and-text-similarity-in-python/
        - https://giovanni.carmantini.com/2016/01/minimum-edit-distance-in-python.html
    """
    size_x = len(source) + 1
    size_y = len(destination) + 1
    matrix = np.zeros([size_x, size_y])
    matrix[:, 0] = np.arange(0, size_x)
    matrix[0, :] = np.arange(0, size_y)
    backtrace = np.zeros_like(matrix, dtype=[('del', bool),
                                             ('sub', bool),
                                             ('ins', bool),
                                             ('cost', int)])
    backtrace[:, 0] = (True, False, False, 0)
    backtrace[0, :] = (False, False, True, 0)

    for x, y in itertools.product(range(1, size_x),
                                  range(1, size_y)):

        if source[x-1] == destination[y-1]:
            cost = 0
        else:
            cost = 1

        delete = matrix[x-1][y] + 1
        insert = matrix[x][y-1] + 1
        substitute = matrix[x-1][y-1] + cost
        min_dist = min(delete, insert, substitute)
        matrix[x, y] = min_dist

        backtrace[x, y] = (delete == min_dist,
                           substitute == min_dist,
                           insert == min_dist,
                           cost)

    return matrix[size_x-1, size_y-1], matrix, backtrace


def naive_backtrace(backtrace: np.ndarray):
    """ Calculate the editing path via the backtrace. """
    rows, columns = backtrace.shape
    i, j = rows-1, columns-1
    backtrace_idndices = [(i, j, 'sub', 0)]

    while (i, j) != (0, 0):
        delete, substitute, insert, cost = backtrace[i, j]
        if insert:
            operation = 'ins'
            i, j = i, j-1

        elif substitute:
            operation = 'sub'
            i, j = i-1, j-1

        elif delete:
            operation = 'del'
            i, j = i - 1, j

        else:
            raise KeyError("Backtrace matrix wrong defined")

        backtrace_idndices.append((i, j, operation, cost))

    return list(reversed(backtrace_idndices))


def decode_(best_path: List[Tuple[int, int, str, int]], source: List[str], destination: List[str]):
    """ Collect all transformations needed to go from `source` to `destination`. """
    to_delete, to_insert, to_substitute = [], [], defaultdict(list)
    for index, (i, j, operation, cost) in enumerate(best_path):

        if operation == 'del':
            item = source[i]
            to_delete.append(item)

        elif operation == 'sub' and cost:   # without cost sub operation indicates correctness
            wrong_item, target_item = source[i], destination[j]
            to_substitute[target_item].append(wrong_item)

        elif operation == 'ins':
            item = destination[j]
            to_insert.append(item)

    return to_delete, to_insert, to_substitute


def update_(confusion_matrix: np.ndarray, to_substitute: dict, alphabet: Alphabet):
    """ Update the confusion matrix. """
    for correct_char, wrong_chars in to_substitute.items():
        correct_char_label = alphabet.label_from_string(correct_char)
        wrong_chars_labels = [alphabet.label_from_string(char) for char in wrong_chars]

        for wrong_char_label in wrong_chars_labels:
            confusion_matrix[correct_char_label, wrong_char_label] += 1


def main(alphabet_path: str, results_path: str, home_directory: str):
    alphabet = Alphabet(alphabet_path)
    inserts, deletes = Counter(), Counter()
    confusion_matrix = np.zeros([alphabet.size, alphabet.size], dtype=int)
    results = pd.read_csv(results_path, usecols=['original', 'prediction'])

    for index, original, prediction in results.itertuples():
        distance, edit_distance_matrix, backtrace = edit_distance(source=prediction,
                                                                  destination=original)
        best_path = naive_backtrace(backtrace)
        to_delete, to_insert, to_substitute = decode_(best_path, prediction, original)
        update_(confusion_matrix, to_substitute, alphabet)
        inserts.update(to_insert)
        deletes.update(to_delete)

    save_confusion_matrix(confusion_matrix, labels=alphabet._label_to_str, directory=home_directory)
    save_donut(inserts, deletes, confusion_matrix, directory=home_directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alphabet', required=True, help='Alphabet as the txt file')
    parser.add_argument('--results', required=True, help='The csv file contains `original` and `prediction` columns')
    parser.add_argument('--home_directory', required=True, help='Directory where save all files')
    parser.add_argument('--log_file', help='Log file')
    parser.add_argument('--log_level', type=int, default=20, help='Log level')
    arguments = parser.parse_args()
    chdir(to='ROOT')

    logger = create_logger(arguments.log_file, level=arguments.log_level, name='confusion_matrix')
    logger.info(f'Arguments: \n{arguments}')
    main(arguments.alphabet, arguments.results, arguments.home_directory)
