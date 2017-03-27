#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Compares bias / unigram distributions."""

from argparse import ArgumentParser
import os

import numpy as np

from emLam.nn.bias import read_prob, read_vocab_arr, read_vocab_map
from emLam.utils import openall


def read_vocab_map(dict_file):
    """Returns the word -> word_index mapping."""
    with openall(dict_file) as inf:
        return {l.split('\t', 1)[0]: i for i, l in enumerate(inf)}


def read_vocab_arr(vocab_file):
    """Reads the word frequency counts into a vector."""
    with openall(vocab_file) as inf:
        return np.array(
            [int(v) for k, v in [l.split('\t') for l in
                                 inf.read().split('\n') if l]],
            dtype=np.float32
        )


def read_prob(prob_file, vocab_map):
    """prob_file should be a tsv with two fields: the log10prob and the word."""
    probs = np.zeros(len(vocab_map), dtype=np.float32)
    with openall(prob_file) as inf:
        lines = map(lambda l: l.strip(), inf)
        lines = dropwhile(lambda l: '\t' not in l, lines)
        for line in takewhile(lambda l: '\t' in l, lines):
            logp, word = line.split('\t')
            word_id = vocab_map.get(word, None)
            if word_id is not None:
                probs[word_id] = np.power(10, np.float32(logp))
            elif word != '<s>':
                raise ValueError('No word if for {}'.format(word))
    return probs


vocab_losses = {
    'raw': vocab_loss,
    'normalized': normalized_vocab_loss,
    'softmax': softmax_vocab_loss,
}


def parse_arguments():
    parser = ArgumentParser(
        description='Tries to approximate the word unigram distribution with '
                    'softmax bias.')
    parser.add_argument('--bias', '-b', required=True, help='the bias file.')
    parser.add_argument('--vocabulary', '-v', required=True,
                        help='the vocabulary file.')
    parser.add_argument('--max-steps', '-m', type=int, default=100,
                        help='the maximum number of iterations [100].')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.1,
                        help='the learning rate [0.1].')
    subparsers = parser.add_subparsers(dest='command',
                                       help='the available approximation '
                                            'methods.')
    vocab_parser = subparsers.add_parser('vocab', help='try to approximate the '
                                                       'word frequencies.')
    vocab_parser.add_argument('--normalize', '-n', choices=vocab_losses.keys(),
                              help='normalize by the word counts for -v.')
    prob_parser = subparsers.add_parser('prob', help='try to approximate the '
                                                     'word probabilities.')
    prob_parser.add_argument('--model-file', '-p',
                             help='the LM model file. If not specified, the ML '
                                  'probabilities are computed from the word '
                                  'frequencies.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    print(args)

    vocab = read_vocab_map(args.vocabulary)
    if args.command == 'vocab':
        target_val = read_vocab_arr(args.vocabulary)
    else:
        if args.model_file:
            target_val = read_prob(args.model_file, vocab)
        else:
            target_val = read_vocab_arr(args.vocabulary)
            target_val /= target_val.sum()

    npz = np.load(args.bias)
    bias_val = npz[next(filter(lambda k: 'softmax_b' in k, npz.keys()))]
    # print("sm_bias\n", sm_bias)

    with tf.Graph().as_default() as graph:
        var_init = tf.constant_initializer(0.01)
        a = tf.get_variable('a', [], tf.float32, initializer=var_init)
        b = tf.get_variable('b', [], tf.float32, initializer=var_init)
        bias_ph = tf.placeholder(tf.float32, shape=bias_val.shape)
        target_ph = tf.placeholder(tf.float32, shape=target_val.shape)
        if args.command == 'vocab':
            loss_fn = vocab_losses[args.normalize]
        else:
            loss_fn = prob_loss
        prediction, loss, = loss_fn(a, b, bias_ph, target_ph)
        train = tf.train.GradientDescentOptimizer(args.learning_rate).minimize(loss)
        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as session:
        session.run(init)
        feed_dict = {bias_ph: bias_val, target_ph: target_val}
        a_val, b_val, pred_val = session.run([a, b, prediction],
                                             feed_dict=feed_dict)
        print('Bias\n', bias_val)
        exp_logits = np.exp(bias_val)
        sm = exp_logits / sum(exp_logits)
        print('SM\n', sm)
        print('pred\n', a_val * sm + b_val)
        print(pred_val, a_val, b_val)

        last_loss = np.inf
        for i in range(args.max_steps):
            a_val, b_val, pred_val, loss_val, _ = session.run(
                [a, b, prediction, loss, train],
                feed_dict=feed_dict
            )
            print(pred_val, a_val, b_val)
            print('Step {}, loss {}, a {}, b {}'.format(i, loss_val, a_val, b_val))
            if loss_val > last_loss or np.abs(last_loss - loss_val) < 1e-5:
                break
            else:
                last_loss = loss_val
        print('Prediction\n', pred_val)
        print('Target\n', target_val)
        full_loss = (pred_val.astype(np.float64) - target_val) ** 2
        print('Full loss\n', np.sqrt(full_loss.mean()))


if __name__ == '__main__':
    main()

