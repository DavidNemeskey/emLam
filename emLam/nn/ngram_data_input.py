#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Data readers that processes ngram count and model files."""

from builtins import filter, map
import logging

import numpy as np

from emLam.utils import openall

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


class NgramModelLoader(object):
    """Gets the data from the model (AT&T format) file."""
    def __init__(self, *args):
        super(NgramModelLoader, self).__init__(*args)


class NgramCountLoader(object):
    """Gets the data from the count file."""
    def __init__(self, *args):
        super(NgramCountLoader, self).__init__(*args)

    def __read_file(self):
        with openall(self.ngram_file, 'rt') as inf:
            data_it = filter(lambda wf: wf[0].count(' ') == self.order - 1,
                             map(lambda l: l.rstrip().split('\t'), inf))
            data = [(word, int(freq)) for word, freq in data]
            
