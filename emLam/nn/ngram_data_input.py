#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Data readers that processes ngram count and model files."""

from builtins import filter, map
from itertools import islice
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


class NgramCountLoaderOld(NgramLoader):
    """Gets the data from the count file."""
    def __init__(self, *args):
        super(NgramCountLoaderOld, self).__init__(*args)
        self.data = self.__read_file()
        self.data_len = sum(freq for _, freq in self.data)
        self.epoch_size = self.data_len // self.batch_size
        self.seed = 42

    def __iter__(self):
        batch_it = self.__random_it()
        data = [[ngram, freq] for ngram, freq in self.data]
        for i in range(self.epoch_size):
            batch_ids = islice(batch_it, self.batch_size)
            batch_ids = self.__transform_batch_ids(batch_ids)
            ngrams = self.__get_ngrams(batch_ids, data)
            yield ngrams

    @staticmethod
    def __transform_batch_ids(batch_ids):
        """
        Transforms the batch ids so that removal (draws without replacement)
        are handled correctly.
        """
        a = np.array(list(batch_ids))
        s = set()
        for i, e in enumerate(a):
            while e in s:
                e += 1
            s.add(e)
            a[i + 1:] = np.where(a[i + 1:] >= e, a[i + 1:] + 1, a[i + 1:])
            a[i] = e
        return sorted(a)

    def __get_ngrams(self, batch_ids, data):
        """Gets the n-grams that correspond to the batch ids."""
        to_delete = []
        ngrams = []
        datait = enumerate(data)
        datai, datap = next(datait)
        curr_freq = datap[1]
        for batch_id in batch_ids:
            while batch_id >= curr_freq:
                datai, datap = next(datait)
                curr_freq += datap[1]
            datap[1] -= 1
            if datap[1] == 0:
                to_delete.append(datai)
            ngrams.append(datap[0])
        for i in to_delete[::-1]:
            data.pop(i)
        return ngrams

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
