#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Prepares data for ngram training."""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from collections import defaultdict, namedtuple
import math

import numpy as np

from emLam.nn.bias import read_vocab_map
from emLam.utils import openall

Probs = namedtuple('Probs', ['word', 'prob'])
Ngrams = namedtuple('Ngrams', ['context', 'word', 'prob'])


def parse_arguments(compilers, formats):
    parser = ArgumentParser(
        description='Prepares data for ngram training.')
    parser.add_argument('ngram_file', help='the AT&T ngram file.')
    parser.add_argument('output_prefix',
                        help='the prefix of the output files\' names. It can '
                             'be a full path, in which case the directory '
                             'structure will be constructed, if needed.')
    parser.add_argument('--order', '-o', type=int, required=True,
                        help='the prefix of the output files\' names. It can '
                             'be a full path, in which case the directory '
                             'structure will be constructed, if needed.')
    parser.add_argument('--vocab-file', '-v', required=True,
                        help='the vocabulary file, created by count_vocab.py. '
                             'Only needed for the int format.')
    return parser.parse_args()


def read_model(ngram_file, order):
    """
    Reads the ngram model; more specifically, the ngram probabilities: backoff
    weights are read by another function.
    """
    header = '\\{}-grams:'.format(order)
    ngrams = []
    last_context = None
    model = defaultdict(dict)
    with openall(ngram_file, 'rt') as inf:
        # TODO read backoff values instead of this
        for line in inf:
            if line.strip() == header:
                break
        for line in inf:
            if '\t' in line:
                fields = line.strip().split('\t')
                context, word = fields[1].rsplit(' ', 1)
                if context != last_context:
                    ngrams.append(Ngrams(context, [], []))
                    last_context = context
                ngrams[-1].word.append(vocab[word])
                ngrams[-1].prob.append(math.pow(10.0, fields[0]))
            else:
                break
    return ngrams


def read_bow(ngram_file, model, order):
    """
    Reads the backoff weights from the model for all the contexts in the model.
    """
    def prepare_order(new_order):
        """Prepares the new order data structures."""
        cu
    bows  = {}
    curr_order, curr_bows = 0, {}
    with openall(ngram_file, 'rt') as inf:
        for line in filter(lambda l: '\t' in l, inf):
            fields = line.strip().split('\t') 
            ngram = fields[1]



def main():
    args = parse_arguments()
    vocab = read_vocab_map(args.vocab_file)
    ngrams = read_model(args.ngram_file, args.order, vocab)


if __name__ == '__main__':
    main()
