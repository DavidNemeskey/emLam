#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Data readers that processes ngram count and model files."""

from builtins import filter, map
import logging
import random

import numpy as np

from emLam.utils import openall
from emLam.nn.bias import read_vocab_map

class NgramLoader(object):
    def __init__(self, ngram_file, order, batch_size, vocab_file,
                 one_hot=False, data_type=np.int32):
        """
        The parameters:
        - ngram_file: the model / count file (see subclasses).
        - order: the ngram order to extract from the file.
        - batch_size: the batch size.
        - vocab_file: for the token -> int mapping.
        """
        super(NgramLoader, self).__init__()
        self.ngram_file = ngram_file
        self.order = order
        self.batch_size = batch_size
        self.vocab_file = vocab_file
        self.one_hot = one_hot
        self.data_type = data_type
        self.vocab = read_vocab_map(vocab_file)
        self.vocab['<s>'] = self.vocab['</s>']  # We don't use <s>


class NgramModelLoader(NgramLoader):
    """Gets the data from the model (AT&T format) file."""
    def __init__(self, *args):
        super(NgramModelLoader, self).__init__(*args)


class NgramCountLoader(NgramLoader):
    """
    Gets the data from the count file. This implementation holds the whole
    index array for the ngrams in memory, but at least it is fast.
    """
    def __init__(self, *args):
        super(NgramCountLoader, self).__init__(*args)
        self.ngrams, self.freqs = self.__read_file()
        self.data_len = self.freqs.sum()
        self.epoch_size = self.data_len // self.batch_size
        self.seed = 42
        self.indices = self.__fill_indices()

    def __fill_indices(self):
        """For the generation."""
        indices = np.zeros(self.data_len, dtype=np.int32)
        last_index = 0
        for i, f in enumerate(self.freqs):
            indices[last_index:last_index + f] = i
            last_index += f
        rnd = np.random.RandomState(self.seed)
        rnd.shuffle(indices)
        return indices

    def __iter__(self):
        last = 0
        for e in range(self.epoch_size):
            yield self.ngrams[self.indices[last:last + self.batch_size]]
            last += self.batch_size

    def __read_file(self):
        with openall(self.ngram_file, 'rt') as inf:
            data_it = filter(lambda ngf: ngf[0].count(' ') == self.order - 1,
                             map(lambda l: l.rstrip().split('\t'), inf))
            data = [(np.array([self.vocab[word] for word in ngram.split(' ')],
                              dtype=np.int32),
                     int(freq)) for ngram, freq in data_it]
            return map(np.array, zip(*data))


class SlowNgramCountLoader(NgramLoader):
    """Gets the data from the count file."""
    def __init__(self, *args):
        super(NgramCountLoader, self).__init__(*args)
        self.data = self.__read_file()
        self.data_len = sum(freq for _, freq in self.data)
        self.epoch_size = self.data_len // self.batch_size
        self.seed = 42

    def __iter__(self):
        aggr_freqs = np.zeros(len(self.data), dtype=np.int32)
        for i in range(len(self.data)):
            aggr_freqs[i] = self.data[i][1] + (aggr_freqs[i - 1] if i > 0 else 0)

        batch_it = self.__random_it()
        for _ in range(self.epoch_size):
            ngrams = [0 for _ in range(self.batch_size)]
            for i in range(self.batch_size):
                batch_id = next(batch_it)
                ins = aggr_freqs.searchsorted(batch_id)
                ngrams[i] = self.data[ins][0]
                aggr_freqs[ins:] -= 1
            yield ngrams

    def __random_it(self):
        """
        Generates random numbers between 0 and the current maximum data size
        (as we draw without replacement).
        """
        rnd = random.Random()
        rnd.seed(self.seed)
        for r in range(self.data_len - 1, -1, -1):
            yield rnd.randint(0, r)

    def __read_file(self):
        with openall(self.ngram_file, 'rt') as inf:
            data_it = filter(lambda ngf: ngf[0].count(' ') == self.order - 1,
                             map(lambda l: l.rstrip().split('\t'), inf))
            data = [(np.array([self.vocab[word] for word in ngram.split(' ')],
                              dtype=np.int32),
                     int(freq)) for ngram, freq in data_it]
            return data
