#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Computes the explicit probability weight present in a certain n-gram order."""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
import math

from emLam.utils import openall


def parse_arguments():
    parser = ArgumentParser(
        description='Computes the explicit probability weight present in a '
                    'certain n-gram order.')
    parser.add_argument('model_file', help='the model file.')
    parser.add_argument('--order', '-o', required=True, type=int,
                        help='the order of the ngrams we are interested in.')
    parser.add_argument('--count-file', '-c',
                        help='the count file for weighting the distributions. '
                             'If not specified, no weighting occurs.')
    return parser.parse_args()


def load_model(ngram_file, order):
    """Loads a specific order of an ngram model."""
    header = '\\{}-grams:'.format(order)
    contexts = []
    ngrams, curr_ngram = [], []
    last_context = None
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
                    if curr_ngram:
                        contexts.append(last_context)
                        ngrams.append(sum(curr_ngram))
                    curr_ngram = []
                    last_context = context
                curr_ngram.append(math.pow(10.0, float(fields[0])))
            else:
                if curr_ngram:
                    ngrams.append(sum(curr_ngram))
                break
    return dict(zip(contexts, ngrams))


def load_counts(count_file, order):
    """Loads the counts for an ngram order."""
    with openall(count_file, 'rt') as inf:
        data_it = filter(lambda ngf: ngf[0].count(' ') == order - 1,
                         map(lambda l: l.rstrip().split('\t'), inf))
        return {ngram: float(count) for ngram, count in data_it}


def main():
    args = parse_arguments()
    model = load_model(args.model_file, args.order)
    if args.count_file:
        weights = load_counts(args.count_file, args.order - 1)
    else:
        weights = {}
    numerator, denominator = 0.0, 0.0
    for context, psum in model.items():
        if context in weights:
            weight = weights.get(context, 1)
        else:
            weight = 1
        numerator += psum * weight
        denominator += weight
    print(numerator / denominator)


if __name__ == '__main__':
    main()
