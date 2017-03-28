#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Converts between RNN bias and unigram LM files."""

from argparse import ArgumentParser

from emLam.nn.bias import read_vocab_map
from emLam.nn.bias import read_bias_file, write_bias_file
from emLam.nn.bias import read_prob, write_prob


def parse_arguments():
    parser = ArgumentParser(
        description='Converts between RNN bias and unigram LM files.')
    parser.add_argument('--vocabulary', '-v', required=True,
                        help='the vocabulary file. Needed so that the word '
                             'order is fixed.')
    parser.add_argument('--bias-file', '-b', required=True, help='the bias file.')
    parser.add_argument('--model-file', '-m', required=True,
                        help='the LM model file.')
    subparsers = parser.add_subparsers(dest='command',
                                       help='the direction of the conversion.')
    subparsers.add_parser('bias2ngram',
                          help='converts from a bias file to unigram LM.')
    subparsers.add_parser('ngram2bias',
                          help='converts from a unigram LM to a bias file.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    vocab_map = read_vocab_map(args.vocabulary)
    if args.command == 'bias2ngram':
        inf, outf = args.bias_file, args.model_file
        infn, outfn = read_bias_file, write_prob
    else:
        inf, outf = args.model_file, args.bias_file
        infn, outfn = read_prob, write_bias_file
    distribution = infn(inf, vocab_map)
    outfn(outf, distribution, vocab_map)


if __name__ == '__main__':
    main()
