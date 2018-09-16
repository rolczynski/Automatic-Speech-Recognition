import numpy as np


class Alphabet:
    """
    Read alphabet.txt, which is the list of valid characters. here are two
    special characters:
      - space: on the beginning
      - blank: default added as the last char

    This class is used to convert characters to labels and vice versa.
    """

    def __init__(self, file_path):
        self.size = 0
        self.blank_token = None
        self._file_path = file_path
        self._label_to_str = []
        self._str_to_label = {}
        self.__process_alphabet_file()


    def __process_alphabet_file(self):
        with open(self._file_path) as file:
            for line in file:
                if line.startswith('#'):
                    continue
                # Char can contain more than one letter
                char = line[:-1]  # remove the line ending
                self._label_to_str += char
                self._str_to_label[char] = self.size
                self.size += 1
            # Blank token is added on the end
            self.blank_token = self.size


    def string_from_label(self, label):
        return self._label_to_str[label]


    def label_from_string(self, string):
        return self._str_to_label[string]


def get_batch_labels(transcripts, alphabet):
    """ Convert batch transcripts to labels """
    batch_labels = [[alphabet.label_from_string(c) for c in transcript]
                    for transcript in transcripts]

    max_len = max(map(len, batch_labels))
    default_value = alphabet.blank_token

    for labels in batch_labels:
        remainder = [default_value] * (max_len-len(labels))
        labels.extend(remainder)

    return np.array(batch_labels)


def get_batch_transcripts(sequences, alphabet):
    """ Convert label sequences to transcripts. The `-1` means the blank tag """
    return [''.join(alphabet.string_from_label(char_label) for char_label in sequence
                    if char_label not in (-1, alphabet.blank_token))
            for sequence in sequences]
