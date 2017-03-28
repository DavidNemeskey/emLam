#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Compares bias / unigram distributions."""

from argparse import ArgumentParser
from itertools import combinations, permutations

import numpy as np
from scipy.stats import entropy

from emLam.nn.bias import read_prob, read_vocab_map, read_bias_file


def parse_arguments():
    parser = ArgumentParser(
        description='Compares bias / unigram distributions.')
    parser.add_argument('--vocabulary', '-v', required=True,
                        help='the vocabulary file. Needed so that the word '
                             'order is fixed.')
    parser.add_argument('--bias', '-b', action='append', default=[],
                        help='a bias file.')
    parser.add_argument('--model-file', '-m', action='append', default=[],
                        help='a LM model file.')
    parser.add_argument('--cost', '-c', choices=['kl', 'js'], default='kl',
                        help='the cost function (KL or JS divergence).')
    args = parser.parse_args()

    if len(args.bias) + len(args.model_file) <= 1:
        parser.error('At least two input files (bias or LM) '
                     'have to be specified.')
    return args


def jsd(p, q):
    """The Jensen-Shannon divergence of two distributions p and q."""
    if p.sum() != 1.0:
        p /= np.linalg.norm(p, ord=1) 
    if q.sum() != 1.0:
        q /= np.linalg.norm(q, ord=1) 
    m = (p + q) / 2
    return 0.5 * (entropy(p, m) + entropy(q, m))


def main():
    args = parse_arguments()

    vocab_map = read_vocab_map(args.vocabulary)
    data = {}
    for input_file in args.bias:
        data[input_file] = read_bias_file(input_file, vocab_map)
    for input_file in args.model_file:
        data[input_file] = read_prob(input_file, vocab_map)

    inputs = sorted(data.keys())
    results = np.zeros((len(inputs), len(inputs)), dtype=np.float32)
    if args.cost == 'js':
        it, fn = combinations, jsd
    else:
        it, fn = permutations, entropy
    for i1, i2 in it(range(len(inputs)), 2):
        results[i1, i2] = fn(data[inputs[i1]], data[inputs[i2]])
    print(inputs)
    np.set_printoptions(suppress=True)
    print(results)


if __name__ == '__main__':
    main()
