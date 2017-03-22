#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Tries to approximate the word unigram distribution with softmax bias."""

from argparse import ArgumentParser
import os

import numpy as np
# To get rid of stupid tensorflow logging. Works from version 0.12.1+.
# See https://github.com/tensorflow/tensorflow/issues/1258
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'  # noqa
import tensorflow as tf

from emLam.utils import openall


def parse_arguments():
    parser = ArgumentParser(
        description='Tries to approximate the word unigram distribution with '
                    'softmax bias.')
    parser.add_argument('--bias', '-b', required=True, help='the bias file.')
    vocab_or_prob = parser.add_mutually_exclusive_group(required=True)
    vocab_or_prob.add_argument('--vocabulary', '-v',
                               help='a vocabulary file. If specified, the '
                                    'script will try to approximate the word '
                                    'frequencies.')
    vocab_or_prob.add_argument('--probability', '-p',
                               help='a word-probability mapping. If specified, '
                                    'the script will try to approximate the '
                                    'word probabilities from the model.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    with openall(args.vocab) as inf:
        vocab = np.array(
            [int(v) for k, v in [l.split('\t') for l in
                                 inf.read().split('\n') if l]],
            dtype=np.float32
        )
    sm_bias = np.load(args.bias)['embedding']
    print("vocab\n", vocab)
    print("sm_bias\n", sm_bias)

    np_pred1 = sm_bias + 1
    np_pred2 = np.exp(np_pred1)
    print('np pred2:\n', np_pred2, '\nnp loss:\n',
          np.power(np_pred2 - vocab, 2).mean(),
          np.power(np_pred1 - np.log(vocab), 2).mean())

    with tf.Graph().as_default() as graph:
        a = tf.get_variable('x', [], tf.float32)
        b = tf.get_variable('y', [], tf.float32)
        bias_ph = tf.placeholder(tf.float32, shape=sm_bias.shape)
        vocab_ph = tf.placeholder(tf.float32, shape=vocab.shape)
        prediction = tf.exp(a*bias_ph + b)
        loss = tf.reduce_mean(tf.squared_difference(prediction / vocab_ph, 1))
        train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as session:
        session.run(init)
        for i in range(30):
            x_val, y_val, pred_val, loss_val, _ = session.run(
                [x, y, prediction, loss, train], feed_dict={b: sm_bias, v: vocab})
            print('Step {}, loss {}, x {}, y {}'.format(i, loss_val, x_val, y_val))
        print("pred\n", pred_val, "exp'd\n", np.exp(pred_val))
        print('np loss:', np.power(x_val * sm_bias + y_val - np.log(vocab), 2).mean())

if __name__ == '__main__':
    main()
