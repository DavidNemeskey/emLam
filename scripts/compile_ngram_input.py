#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Prepares data for ngram training."""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
import json
import math

from emLam.nn.bias import read_vocab_map
from emLam.utils import openall


# TODO: frequencies


class Ngrams(object):
    def __init__(self, context, word=None, prob=None, bow=0):
        self.context = context
        self.word = word or []
        self.prob = prob or []
        self.bow = bow

    def __repr__(self):
        return '{}\t{}\t{}'.format(self.context, list(zip(self.word, self.prob)), self.bow)


def parse_arguments():
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


def read_ngrams(ngram_file, order, keep=None):
    """Reads ngrams of a certain order from file."""
    keep = keep or set()
    header = '\\{}-grams:'.format(order)
    with openall(ngram_file, 'rt') as inf:
        for line in inf:
            if line.strip() == header:
                break
        for line in inf:
            if '\t' in line:
                fields = line.strip().split('\t')
                prob = float(fields[0])
                ngram = fields[1]
                bow = float(fields[2]) if len(fields) > 2 else None
                if not keep or ngram in keep:
                    yield prob, ngram, bow
            else:
                break


def ngrams_to_model(ngram_stream):
    """Consumes an ngram stream and convert it to a model."""
    ngrams = []  # We drop the first, see below
    curr_context = None
    for prob, ngram, bow in ngram_stream:
        context, word = ngram.rsplit(' ', 1)
        if context != curr_context:
            ngrams.append(Ngrams(context))
            curr_context = context
        ngrams[-1].word.append(word)
        ngrams[-1].prob.append(prob)
    return ngrams


def read_bows(ngram_file, model, order):
    """
    Reads the backoff weights from the model for all the contexts in the model.
    It is enough to get the backoff weight for the full context, as we will
    compare our results against the n-1 gram probabilities, which already
    include the rest of the BOWs.

    This method is fail-fast: it assumes that all contexts have BOWs in the
    lower order, and fails with an exception if not.
    """
    contexts = read_ngrams(ngram_file, order - 1,
                           keep=set(ngram.context for ngram in model))
    bows = {context: bow for _, context, bow in contexts}
    for ngram in model:
        ngram.bow = bows[ngram.context]


def words_to_ids(model, vocab):
    """Converts the model to use lists of word ids from vocab."""
    return [
        ([vocab[cword] for cword in ngram.context.split(' ')],
         {'words': [vocab[word] for word in ngram.word],
          'probs': [math.pow(10, prob) for prob in ngram.prob],
          'bow': math.pow(10, ngram.bow)})
        for ngram in model
    ]


def read_vocab(vocab_file):
    """
    Bridges the difference between the n-gram model, which use <s>, and the
    neural one, that does not.
    """
    vocab = read_vocab_map(vocab_file)
    vocab['<s>'] = vocab['</s>']
    return vocab


def main():
    args = parse_arguments()
    ngrams = ngrams_to_model(read_ngrams(args.ngram_file, args.order))
    read_bows(args.ngram_file, ngrams, args.order)
    model = words_to_ids(ngrams, read_vocab(args.vocab_file))
    with openall(args.output_prefix + '.gz', 'wt') as outf:
        json.dump(model, outf)


if __name__ == '__main__':
    main()
