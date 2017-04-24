#!/usr/bin/env python

from scipy.io.matlab import loadmat
import os
import numpy as np
import copy
from collections import OrderedDict
import theano
import theano.tensor as T
import logging

# FORMAT = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
# _log = logging.getLogger(__name__)
# _log.setLevel(logging.DEBUG)
# ch_file = logging.FileHandler(filename='my.log', mode='w')
# ch_file.setLevel(logging.DEBUG)
# ch_file.setFormatter(FORMAT)
# ch = logging.StreamHandler()
# ch.setFormatter(FORMAT)
# ch.setLevel(logging.DEBUG)
# _log.addHandler(ch)
# _log.addHandler(ch_file)

import math
import time
import pickle


class DD(dict):
    """This class is only used to replace a state variable of Jobman"""

    def __getattr__(self, attr):
        if attr == '__getstate__':
            return super(DD, self).__getstate__
        elif attr == '__setstate__':
            return super(DD, self).__setstate__
        elif attr == '__slots__':
            return super(DD, self).__slots__
        return self[attr]

    def __setattr__(self, attr, value):
        assert attr not in ('__getstate__', '__setstate__', '__slots__')
        self[attr] = value

    def __str__(self):
        return 'DD%s' % dict(self)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memo):
        z = DD()
        for k, kv in self.iteritems():
            z[k] = copy.deepcopy(kv, memo)
        return z


# ----------------------------------------------------------------------------

# Mappings class -----------------------------------------------------------
class Mappings(object):
    """Class for the embeddings matrix."""

    def __init__(self, rng, N, D, tag=''):
        """
        Constructor.

        :param rng: numpy.random module for number generation.
        :param N: number of the features.
        :param D: dimension of the reduced dimensionality.
        :param tag: name of the mapping matrix.
        """
        self.N = N
        self.D = D
        wbound = np.sqrt(6. / D)
        W_values = rng.uniform(low=-wbound, high=wbound, size=(D, N))
        W_values = W_values / np.sqrt(np.sum(W_values ** 2, axis=0))  # Columns are normalized
        W_values = np.asarray(W_values, dtype=theano.config.floatX)
        self.E = theano.shared(value=W_values, name='E' + tag)
        # Define a normalization function with respect to the L_2 norm of the
        # mapping vectors.
        self.updates = OrderedDict({self.E: self.E / T.sqrt(T.sum(self.E ** 2, axis=0))})
        self.normalize = theano.function([], [], updates=self.updates)


# ----------------------------------------------------------------------------

class Mappings1(object):
    """Class for the embeddings matrix."""

    def __init__(self, rng, N, D, tag=''):
        """
        Constructor.

        :param rng: numpy.random module for number generation.
        :param N: number of the features.
        :param D: dimension of the reduced dimensionality.
        :param tag: name of the mapping matrix.
        """
        self.N = N
        self.D = D
        W_values = 1. * rng.random(size=(D, N))
        W_values = np.asarray(W_values, dtype=theano.config.floatX)
        self.E = theano.shared(value=W_values, name='E' + tag)
        # Define a normalization function with respect to the L_2 norm of the
        # mapping vectors.
        self.updates = OrderedDict({self.E: self.E / T.sqrt(T.sum(self.E ** 2, axis=0))})
        self.normalize = theano.function([], [], updates=self.updates)


# ----------------------------------------------------------------------------

def softmax(Y):
    """
    turn the distances into probability using the softmax function
    :param X: (M x N) data matrix
    :param A: (N x K) projection matrix
    :return: (M x M) probability matrix about the neighbourhood likeness
    """
    sumY = T.sum(T.sqr(Y), axis=1)  # column sum, sumY.shape = (M,)
    Dist = sumY.dimshuffle('x', 0) + sumY.dimshuffle(0, 'x') - 2 * Y.dot(Y.T)
    expDist = T.exp(-Dist)
    expDist = T.fill_diagonal(expDist, 0)
    return expDist / T.sum(expDist, axis=1)  # (M, M) / (M,) : column sum is one


def softcauchy(Y):
    """
     turn the distances into probability using the cauchy distribution
    :param X: X: (M x N) data matrix
    :param A: (N x K) projection matrix
    :return: (M x M) probability matrix about the neighbourhood likeness
    """
    sumY = T.sum(T.sqr(Y), axis=1)  # column sum, sumY.shape = (M,)
    dist = sumY.dimshuffle('x', 0) + sumY.dimshuffle(0, 'x') - 2 * Y.dot(Y.T)
    cauchy_dist = 1 / (dist + 1)
    cauchy_dist = T.fill_diagonal(cauchy_dist, 0)
    return cauchy_dist / T.sum(cauchy_dist, axis=1)  # (M, M) / (M,) : column sum is one. Pr(j,i)


# ----------------------------------------------------------------------------

def margincost(pos, neg, marge=.1):
    out = neg - pos + marge
    return T.mean(out * (out > 0)), out > 0


# ----------------------------------------------------------------------------

def TrainFnMember(fnPr, X, mapping, marge):             # standard margin-based cost
    """

    :param fnPr:
    :param mapping:
    :param marge:
    :return:
    """

    # define the required symbolic input
    inpl, inpr, inprn = T.ivectors(3)
    lrmapping = T.scalar('lrmapping')  # learning rate

    list_in = [inpl, inpr, inprn, lrmapping]  # highlighting input argument

    Pr = fnPr(X.dot(mapping.E.T))  # mapping.E (K x N) matrix
    p = Pr[inpr, inpl]  # (L,)
    pln = Pr[inprn, inpl]
    cost, out = margincost(p, pln, marge)

    # assumming no other parameters
    gradients_mapping = T.grad(cost, mapping.E)
    newE = mapping.E - lrmapping * gradients_mapping
    updates = OrderedDict({mapping.E: newE})

    """
        Theano function inputs.
        :input lrmapping: learning rate for the mapping matrix.
        :input inpl: vector representing the indexes of the positive
                     citations 'left' member, shape=(#examples,).
        :input inpr: vector representing the indexes of the positive
                     citations 'right' member, shape=(#examples,).
        :input inpln: vector representing the indexes of the negative
                     citations 'left' member, shape=(#examples,).

        Theano function output.
        :output mean(cost): average cost.
        :output mean(out): ratio of examples for which the margin is violated,
                           i.e. for which an update occurs.
        """
    return theano.function(list_in, [T.mean(cost), T.mean(out), T.mean(p)],
                           updates=updates, on_unused_input='ignore')


# ----------------------------------------------------------------------------

def TrainFn1Member(fnPr, X, mapping):
    """

    :param fnPr:
    :param mapping:
    :return:
    """
    # define the required symbolic input
    inpl, inpr = T.ivectors(2)
    lrmapping = T.scalar('lrmapping')  # learning rate

    list_in = [inpl, inpr, lrmapping]  # highlighting input argument

    Pr = fnPr(X.dot(mapping.E.T))  # mapping.E (K x N) matrix
    p = Pr[inpr, inpl]  # (L,)
    cost = -T.mean(p)

    # assumming no other parameters
    gradients_mapping = T.grad(cost, mapping.E)
    newE = mapping.E - lrmapping * gradients_mapping
    updates = OrderedDict({mapping.E: newE})
    """
        Theano function inputs.
        :input lrmapping: learning rate for the mapping matrix.
        :input inpl: vector representing the indexes of the positive
                     citations 'left' member, shape=(#examples,).
        :input inpr: vector representing the indexes of the positive
                     citations 'right' member, shape=(#examples,).
        :input inpln: vector representing the indexes of the negative
                     citations 'left' member, shape=(#examples,).

        Theano function output.
        :output mean(cost): average cost.
        :output mean(out): ratio of examples for which the margin is violated,
                           i.e. for which an update occurs.
        """
    return theano.function(list_in, [cost],
                           updates=updates, on_unused_input='ignore')


# ----------------------------------------------------------------------------

def TrainFn2Member(fnPr, X):
    """

    :param fnPr:
    :param mapping:
    :return:
    """
    # define the required symbolic input
    inpl, inpr = T.ivectors(2)
    A = T.vector()
    k = T.iscalar()

    list_in = [A, inpl, inpr, k]  # highlighting input argument

    Pr = fnPr(X.dot(A.reshape((X.shape[1], k), ndim=2)))  # matrix A (N x K) matrix
    p = Pr[inpr, inpl]  # (L,)
    cost = -T.mean(p)

    # assumming no other parameters
    # gradients_mapping = T.grad(cost, A)

    """
        Theano function inputs.
        :input lrmapping: learning rate for the mapping matrix.
        :input inpl: vector representing the indexes of the positive
                     citations 'left' member, shape=(#examples,).
        :input inpr: vector representing the indexes of the positive
                     citations 'right' member, shape=(#examples,).
        :input inpln: vector representing the indexes of the negative
                     citations 'left' member, shape=(#examples,).

        Theano function output.
        :output mean(cost): average cost.
        :output mean(out): ratio of examples for which the margin is violated,
                           i.e. for which an update occurs.
        """
    return theano.function(list_in, cost, on_unused_input='ignore',
                           allow_input_downcast=True), theano.function(list_in,
                                                                       T.grad(cost, A), allow_input_downcast=True)


# ----------------------------------------------------------------------------


def TrainFn3Member(fnPr, mapping, marge):
    """

    :param fnPr:
    :param mapping:
    :param marge:
    :return:
    """

    # define the required symbolic input
    inpl, inpr, inprn = T.ivectors(3)
    lrmapping = T.scalar('lrmapping')  # learning rate

    list_in = [inpl, inpr, inprn, lrmapping]  # highlighting input argument

    Pr = fnPr(mapping.E.T)  # mapping.E ( K x M) matrix
    p = Pr[inpr, inpl]  # (L,)
    pln = Pr[inprn, inpl]
    cost, out = margincost(p, pln, marge)

    # assumming no other parameters
    gradients_mapping = T.grad(cost, mapping.E)
    newE = mapping.E - lrmapping * gradients_mapping
    updates = OrderedDict({mapping.E: newE})

    """
        Theano function inputs.
        :input lrmapping: learning rate for the mapping matrix.
        :input inpl: vector representing the indexes of the positive
                     citations 'left' member, shape=(#examples,).
        :input inpr: vector representing the indexes of the positive
                     citations 'right' member, shape=(#examples,).
        :input inpln: vector representing the indexes of the negative
                     citations 'left' member, shape=(#examples,).

        Theano function output.
        :output mean(cost): average cost.
        :output mean(out): ratio of examples for which the margin is violated,
                           i.e. for which an update occurs.
        """
    return theano.function(list_in, [T.mean(cost), T.mean(out), T.mean(p)],
                           updates=updates, on_unused_input='ignore')


# ----------------------------------------------------------------------------

def TrainFn4Member(fnPr,  mapping, sigma, marge):             # standard margin-based cost
    """

    :param fnPr:
    :param mapping:
    :param marge:
    :return:
    """

    # define the required symbolic input
    inpl, inpr, inprn = T.ivectors(3)
    lrmapping = T.scalar('lrmapping')  # learning rate
    lrparam = T.scalar()

    list_in = [inpl, inpr, inprn, lrmapping, lrparam]  # highlighting input argument

    Pr = fnPr(mapping.E.T / sigma)  # mapping.E (K x N) matrix
    p = Pr[inpr, inpl]  # (L,)
    pln = Pr[inprn, inpl]
    cost, out = margincost(p, pln, marge)

    # assumming no other parameters
    gradients_mapping = T.grad(cost, mapping.E)
    newE = mapping.E - lrmapping * gradients_mapping
    updates = OrderedDict({mapping.E: newE})

    new_sigma = sigma - lrparam * sigma
    updates.update({sigma: new_sigma})

    """
        Theano function inputs.
        :input lrmapping: learning rate for the mapping matrix.
        :input inpl: vector representing the indexes of the positive
                     citations 'left' member, shape=(#examples,).
        :input inpr: vector representing the indexes of the positive
                     citations 'right' member, shape=(#examples,).
        :input inpln: vector representing the indexes of the negative
                     citations 'left' member, shape=(#examples,).

        Theano function output.
        :output mean(cost): average cost.
        :output mean(out): ratio of examples for which the margin is violated,
                           i.e. for which an update occurs.
        """
    return theano.function(list_in, [T.mean(cost), T.mean(out), T.mean(p)],
                           updates=updates, on_unused_input='ignore')

# ----------------------------------------------------------------------------

def RankFnIdx(fnPr, X, mapping):
    """

    :param fnPr:
    :param X:
    :param mapping:
    :return:
    """

    Pr = fnPr(X, mapping.E.T)  # mapping.E (K x N) matrix

    """
        Theano function inputs.
        :input lrmapping: learning rate for the mapping matrix.
        :input inpl: vector representing the indexes of the positive
                     citations 'left' member, shape=(#examples,).
        :input inpr: vector representing the indexes of the positive
                     citations 'right' member, shape=(#examples,).
        :input inpln: vector representing the indexes of the negative
                     citations 'left' member, shape=(#examples,).

        Theano function output.
        :output mean(cost): average cost.
        :output mean(out): ratio of examples for which the margin is violated,
                           i.e. for which an update occurs.
        """
    return theano.function([], [Pr],
                           on_unused_input='ignore')


# ----------------------------------------------------------------------------

def RankScoreIdx(Pr, idxl, idxr):
    err = []
    for l, r in zip(idxl, idxr):
        err += [np.argsort(np.argsort(
            Pr[:, l])[::-1])[r] + 1]
    return err
