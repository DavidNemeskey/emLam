#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Bias-related functions."""

from itertools import dropwhile, takewhile

import numpy as np

from emLam.utils import openall


def read_vocab_map(dict_file):
    """Returns the word -> word_index mapping."""
    with openall(dict_file) as inf:
        return {l.split('\t', 1)[0]: i for i, l in enumerate(inf)}


def read_vocab_arr(vocab_file):
    """Reads the word frequency counts into a vector."""
    with openall(vocab_file) as inf:
        return np.array(
            [int(v) for k, v in [l.split('\t') for l in
                                 inf.read().split('\n') if l]],
            dtype=np.float32
        )


def read_prob(prob_file, vocab_map):
    """
    Reads a (unigram) probability distribution. prob_file could be either a tsv
    with two fields: the log10prob and the word, or an AT&T LM file (from
    SRILM).
    """
    probs = np.zeros(len(vocab_map), dtype=np.float32)
    with openall(prob_file) as inf:
        lines = map(lambda l: l.strip(), inf)
        lines = dropwhile(lambda l: '\t' not in l, lines)
        for line in takewhile(lambda l: '\t' in l, lines):
            logp, word = line.split('\t')
            word_id = vocab_map.get(word, None)
            if word_id is not None:
                probs[word_id] = np.power(10, np.float32(logp))
            elif word != '<s>':
                raise ValueError('No word if for {}'.format(word))
    return probs


def read_bias_file(bias_file, vocab_map):
    """Reads a bias file saved from a model and and loads it to a np array."""
    npz = np.load(bias_file)
    vocab = npz['vocab']
    sorter = np.argsort([vocab_map[w] for w in vocab])
    bias = npz[next(filter(lambda k: 'softmax_b' in k, npz.keys()))][sorter]
    exp_logits = np.exp(bias)
    return exp_logits / np.sum(exp_logits)
