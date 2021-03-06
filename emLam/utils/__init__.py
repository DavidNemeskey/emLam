#!/usr/bin/env python3
"""Generic utility functions."""

from __future__ import absolute_import, division, print_function
from builtins import map
from past.builtins import basestring
from contextlib import contextmanager
import bz2
from functools import partial
import gzip
from io import open, TextIOWrapper
import logging
from multiprocessing import Manager, Pool
import os
import os.path as op
from queue import Queue
import re
import sys

from emLam.utils.queue_handler import QueueListener, QueueHandler

__allname_p = re.compile(r'^(.+?)(\.gz|\.bz2)?$')

if sys.version_info.major == 3:
    def openall(
        filename, mode='rt', encoding=None, errors=None, newline=None,
        buffering=-1, closefd=True, opener=None,  # for open()
        compresslevel=5,  # faster default compression
    ):
        """
        Opens all file types known to the Python SL. There are some differences
        from the stock functions:
        - the default mode is 'rt'
        - the default compresslevel is 5, because e.g. gzip does not benefit a lot
          from higher values, only becomes slower.
        """
        if filename.endswith('.gz'):
            return gzip.open(filename, mode, compresslevel,
                             encoding, errors, newline)
        elif filename.endswith('.bz2'):
            return bz2.open(filename, mode, compresslevel,
                            encoding, errors, newline)
        else:
            return open(filename, mode, buffering, encoding, errors, newline,
                        closefd, opener)
else:
    class PatchedGzipFile(gzip.GzipFile):
        """
        GzipFile in Python 2.7 does not implement read1(), which is needed by
        TextIOWrapper.  Good, consistent API design! :/
        See https://bugs.python.org/issue10791.
        """
        def read1(self, n):
                return self.read(n)

    def openall(
        filename, mode='rt', encoding=None, errors=None, newline=None,
        buffering=-1, closefd=True, opener=None,  # for open()
        compresslevel=5,  # faster default compression
    ):
        """
        Opens all file types known to the Python SL. There are some differences
        from the stock functions:
        - the default mode is 'rt'
        - the default compresslevel is 5, because e.g. gzip does not benefit a lot
          from higher values, only becomes slower.
        """
        if filename.endswith('.gz'):
            f = PatchedGzipFile(filename, mode, compresslevel)
        elif filename.endswith('.bz2'):
            f = bz2.open(filename, mode, compresslevel)
        else:
            return open(filename, mode, buffering, encoding, errors, newline,
                        closefd)
        return TextIOWrapper(f, encoding, errors, newline)


def allname(fn):
    """
    Returns the "base" name and the extension of a file separately, Only the
    extensions openall() handles are considered.
    """
    return __allname_p.match(fn).groups()


def walk_non_hidden(directory):
    """Walks directory as os.walk, skipping hidden files and directories."""
    def delete_hidden(lst):
        for i in range(len(lst) - 1, -1, -1):
            if lst[i][0] == '.':
                del lst[i]

    for tup in os.walk(directory):
        dirpath, dirnames, filenames = tup
        delete_hidden(dirnames)
        delete_hidden(filenames)
        yield tup


def commonpath(paths):
    """Same as os.path.commonpath, but it is only available from 3.5."""
    common = op.commonprefix(paths)
    while common[-1] != os.sep:
        common = common[:-1]
    return common


def source_target_file_list(source, target_dir):
    """
    Based on the source and target directory provided, returns
    source-target file pairs.
    """
    source = op.abspath(source)
    target_dir = op.abspath(target_dir)

    # Directory source
    if op.isdir(source):
        source_files = [op.abspath(op.join(d, f))
                        for d, _, fs in walk_non_hidden(source) for f in fs]
        source_dir = source
    # File list source
    elif op.isfile(source):
        with openall(source) as inf:
            source_files = [op.abspath(p) for p in inf.read().split()]
        # To be able to keep the directory structure, if any. If there is no
        # common path prefix, the target directory will be flat (see below)
        source_dir = commonpath(source_files)
        if source_dir == '/':  # FIXME: no idea how this works on Windows
            source_dir = ''
    else:
        raise ValueError('Source {} does not exist.'.format(source))

    target_files = []
    for sf in source_files:
        if source_dir:
            sf_rel = sf[len(source_dir):].lstrip(os.sep)  # Structure
        else:
            sf_rel = op.basename(sf)  # Flat
        tf = op.join(target_dir, sf_rel)
        td = op.dirname(tf)
        if not op.isdir(td):
            os.makedirs(td)
        target_files.append(tf)
    return list(zip(source_files, target_files))


def get_any_index(lst, *values):
    """Returns the index of (the first of) a set of values in lst."""
    for value in values:
        try:
            return lst.index(value)
        except ValueError:
            pass


@contextmanager
def __configure_logging(fn, processes, logging_level, **kwargs):
    if logging_level:
        log_level = __get_logging_level(logging_level)
        logging_queue = Queue() if processes == 1 else Manager().Queue()
        sh = logging.StreamHandler()
        sh.setLevel(log_level)
        process_log = '(%(process)d)' if processes > 1 else ''
        f = logging.Formatter(
            '%(asctime)s %(levelname)s %(name)s{} %(message)s'.format(process_log))
        sh.setFormatter(f)
        ql = QueueListener(logging_queue, sh)
        ql.start()
        f = partial(fn, logging_level=log_level,
                    logging_queue=logging_queue, **kwargs)
    else:
        f = partial(fn, **kwargs)
        ql = None

    yield f

    if ql:
        ql.stop()


def run_function(fn, params, processes=1, logging_level=None):
    """
    Runs fn with parameters params. If the number of processes is 1, it is
    equivalent to fn(params); otherwise, the function is run in a
    multiprocessing pool.
    """
    if processes < 1:
        raise ValueError('Number of processes must be at least 1.')

    with __configure_logging(fn, processes, logging_level) as f:
        if processes == 1:
            return list(map(f, params))
        else:
            p = Pool(processes)
            ret = p.map(f, params)
            p.close()
            p.join()
            return ret


def run_queued(fn, params, processes=1, queued_params=None, logging_level=None):
    """
    Same as run_function, but the function should be such that it accepts a
    parameter queue and reads its inputs from there; params still contains
    options for the function.

    If logging_level is not None, a QueueHandler -- QueueListener pair is set
    up so that the function can forward its logging to the stderr of the main
    process. In this case, the params sent to the function are extended by
    this logging level and the logging queue.
    """
    if processes < 1:
        raise ValueError('Number of processes must be at least 1.')

    queue = Queue() if processes == 1 else Manager().Queue()
    for qp in queued_params:
        queue.put_nowait(qp)

    with __configure_logging(fn, processes, logging_level, queue=queue) as f:
        # f = partial(fn, queue=queue)
        if processes == 1:
            ret = list(map(f, params))
        else:
            p = Pool(processes)
            ret = p.map(f, params)
            p.close()
            p.join()

    return ret


def setup_queue_logger(logging_level, logging_queue, name='script'):
    """Sets a queue logger up."""
    qh = QueueHandler(logging_queue)
    return setup_logger(logging_level, qh, name)


def setup_stream_logger(logging_level, name='script'):
    """Sets a stream logger up."""
    sh = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)
    return setup_logger(logging_level, sh, name)


def __get_logging_level(logging_level):
    """Returns the logging level that corresponds to the parameter string."""
    if isinstance(logging_level, basestring):
        return getattr(logging, logging_level.upper())
    else:
        return logging_level


def setup_logger(logging_level, handler, name='script'):
    """Setups logging for scripts."""
    logger = logging.getLogger('emLam')
    # Remove old handlers
    while logger.handlers:
        logger.removeHandler(logger.handlers[-1])

    if logging_level:
        log_level = __get_logging_level(logging_level)
        # Set up root logger
        handler.setLevel(log_level)
        logger.addHandler(handler)
    else:
        # Don't log anything
        log_level = logging.CRITICAL + 1
    logger.setLevel(log_level)

    # Set up the specific logger requested
    logger = logging.getLogger('emLam.' + name)
    logger.setLevel(log_level)
    return logger


class AttrDict(dict):
    """Makes our life easier."""
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError('key {} missing'.format(key))
        return self[key]

    def __setattr__(self, key, value):
        if key not in self:
            raise AttributeError('key {} missing'.format(key))
        self[key] = value


def read_conll(instream):
    """A generator that returns the sentences from a conll file."""
    sentence = []
    for line in instream:
        l = line.strip()
        if len(l) == 0:  # EoS
            yield sentence
            sentence = []
        else:
            sentence.append(l.split(u'\t'))
    else:
        if len(sentence) > 0:
            yield sentence


def write_conll(sentence, outstream):
    print(u'\n'.join(u'\t'.join(word) for word in sentence) + '\n',
          file=outstream)
