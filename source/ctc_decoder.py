"""
Author: Awni Hannun
This is an example CTC decoder written in Python. The code is
intended to be a simple example and is not designed to be
especially efficient.
The algorithm is a prefix beam search for a model trained
with the CTC loss function.
For more details checkout either of these references:
  https://distill.pub/2017/ctc/#inference
  https://arxiv.org/abs/1408.2873
"""

import numpy as np
import math
import collections
NEG_INF = -float("inf")


def batch_decode(y_hat, alphabet):
    """ Enable to batch decode using multiprocessing """
    raise NotImplemented


def __decode(probs, alphabet, beam_size=100, prune=0.001):
    """
    Performs inference for the given output probabilities.
    Arguments:
        probs: The output probabilities (e.g. post-softmax) for each
          time step. Should be an array of shape (time x output dim).
        beam_size (int): Size of the beam to use during inference.
        blank (int): Index of the CTC blank label.
    Returns the output label sequence and the corresponding negative
    log-likelihood estimated by the decoder.
    """
    # Add the blank token to the alphabet
    alphabet._label_to_str.append('')
    alphabet._str_to_label[''] = 35

    T, S = probs.shape
    probs, prune = np.log(probs), np.log(prune)

    # Elements in the beam are (prefix, (p_blank, p_no_blank))
    # Initialize the beam with the empty sequence, a probability of
    # 1 for ending in blank and zero for ending in non-blank
    # (in log space).
    beam = [('', (0.0, NEG_INF))]

    for t in range(T):  # Loop over time

        # A default dictionary to store the next step candidates.
        next_beam = __make_new_beam()

        pruned_indices, = np.where(probs[t] > prune)
        pruned_alphabet = [(i, alphabet.string_from_label(i))
                           for i in pruned_indices]

        for i, s in pruned_alphabet:  # Loop over most probable characters
            P = probs[t, i]

            # The variables P_b and P_nb are respectively the
            # probabilities for the prefix given that it ends in a
            # blank and does not end in a blank at this time step.
            for prefix, (P_b, P_nb) in beam:  # Loop over beam

                # If we propose a blank the prefix doesn't change.
                # Only the probability of ending in blank gets updated.
                if s == '':
                    next_P_b, next_P_nb = next_beam[prefix]
                    next_P_b = __logsumexp(next_P_b, P_b + P, P_nb + P)
                    next_beam[prefix] = (next_P_b, next_P_nb)
                    continue

                # Extend the prefix by the new character s and add it to
                # the beam. Only the probability of not ending in blank
                # gets updated.
                end_t = prefix[-1] if prefix else None
                next_prefix = prefix + s
                next_P_b, next_P_nb = next_beam[next_prefix]
                if s != end_t:
                    next_P_nb = __logsumexp(next_P_nb, P_b + P, P_nb + P)
                else:
                    # We don't include the previous probability of not ending
                    # in blank (P_nb) if s is repeated at the end. The CTC
                    # algorithm merges characters not separated by a blank.
                    next_P_nb = __logsumexp(next_P_nb, P_b + P)

                # *NB* this would be a good place to include an LM score.
                next_beam[next_prefix] = (next_P_b, next_P_nb)

                # If s is repeated at the end we also update the unchanged
                # prefix. This is the merging case.
                if s == end_t:
                    next_P_b, next_P_nb = next_beam[prefix]
                    next_P_nb = __logsumexp(next_P_nb, P_nb + P)
                    next_beam[prefix] = (next_P_b, next_P_nb)

        # Sort and trim the beam before moving on to the
        # next time-step.
        beam = sorted(next_beam.items(),
                      key=lambda x: __logsumexp(*x[1]),
                      reverse=True)
        beam = beam[:beam_size]

    best = beam[0]
    return best[0], -__logsumexp(*best[1])


def __make_new_beam():
    fn = lambda: (NEG_INF, NEG_INF)
    return collections.defaultdict(fn)


def __logsumexp(*args):
    """
    Stable log sum exp.
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp