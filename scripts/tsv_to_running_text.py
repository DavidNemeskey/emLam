#!/usr/bin/env python3
"""
Converts the tsv format to running text (SRILM-style, one sentence per line).
"""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from functools import partial
import os

from emLam import WORD, LEMMA
from emLam.conversions import get_field_function, list_field_functions
from emLam.utils import openall, run_function, source_target_file_list


def parse_arguments():
    parser = ArgumentParser(
        description='Converts the tsv format to running text (SRILM-style, one '
                    'sentence per line).')
    parser.add_argument('--source-dir', '-s', required=True,
                        help='the source directory.')
    parser.add_argument('--target-dir', '-t', required=True,
                        help='the target directory.')
    parser.add_argument('--lower', '-l', action='store_true',
                        help='lowercase the word and lemma.')
    parser.add_argument('--field', '-f', default=None,
                        choices=sorted(list_field_functions()),
                        help='the field section function.')
    parser.add_argument('--processes', '-p', type=int, default=1,
                        help='the number of processes to use parallelly [1].')

    args = parser.parse_args()
    if args.source_dir == args.target_dir:
        parser.error('Source and target directories must differ.')
    if args.processes <= 0:
        parser.error('The number of processes must be >= 1.')
    return args


def convert(source_target_file, field_fun, lowercase):
    source_file, target_file = source_target_file
    with openall(source_file) as inf, openall(target_file, 'wt') as outf:
        sentence = []
        for line in inf:
            line = line.strip()
            if len(line) > 0:
                fields = line.split("\t")
                if lowercase:
                    fields[WORD] = fields[WORD].lower()
                    fields[LEMMA] = fields[LEMMA].lower()
                sentence.extend(field_fun(fields))
            else:
                if len(sentence) > 0:
                    print(u' '.join(sentence), file=outf)
                    sentence = []
        if len(sentence) > 0:
            print(u' '.join(sentence), file=outf)


def main():
    args = parse_arguments()
    field_fun = get_field_function(args.field)
    if not os.path.isdir(args.target_dir):
        os.makedirs(args.target_dir)

    files = source_target_file_list(args.source_dir, args.target_dir)
    fn = partial(convert, field_fun=field_fun, lowercase=args.lower)
    run_function(fn, files, args.processes)


if __name__ == '__main__':
    main()
