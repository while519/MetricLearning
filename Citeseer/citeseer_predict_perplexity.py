#!/usr/bin/env python

from scipy.io.matlab import loadmat
from scipy.io.matlab import savemat
import os
import numpy as np
import copy
from collections import OrderedDict
import theano
import theano.tensor as T
import logging
import math
import time
import pickle

dataname = 'citeseer'
applyfn = 'softcauchy'
outdim = 20
FORMAT = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
_log = logging.getLogger(dataname +' experiment')
_log.setLevel(logging.DEBUG)
ch_file = logging.FileHandler(filename='predict_perplexity_' + applyfn + str(outdim) + '.log', mode='w')
ch_file.setLevel(logging.DEBUG)
ch_file.setFormatter(FORMAT)
ch = logging.StreamHandler()
ch.setFormatter(FORMAT)
ch.setLevel(logging.DEBUG)
_log.addHandler(ch)
_log.addHandler(ch_file)


# ----------------------------------------------------------------------------
def Hbeta(D=np.array([]), beta=1.0):
    """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP      # Shannon Entropy
    P = P / sumP
    return H, P


# ----------------------------------------------------------------------------
def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point ", i, " of ", n, "...")

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2;

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries = tries + 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    return P
# ----------------------------------------------------------------------------
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
        W_values = W_values / np.sqrt(np.sum(W_values ** 2))  # Columns are normalized
        W_values = np.asarray(W_values, dtype=theano.config.floatX)
        self.E = theano.shared(value=W_values, name='E' + tag)
        # Define a normalization function with respect to the L_2 norm of the
        # mapping vectors.
        self.updates = OrderedDict({self.E: self.E / T.sqrt(T.sum(self.E ** 2, axis=0))})
        self.normalize = theano.function([], [], updates=self.updates)


# ----------------------------------------------------------------------------
def softmax(Y):
    """
    turn the distances into probability using the safe softmax function
    :param X: (M x N) data matrix
    :param A: (N x K) projection matrix
    :return: (M x M) probability matrix about the neighbourhood likeness
    """
    sumY = T.sum(T.sqr(Y), axis=1)  # column sum, sumY.shape = (M,)
    Dist = sumY.dimshuffle('x', 0) + sumY.dimshuffle(0, 'x') - 2 * Y.dot(Y.T)
    max_Dist = T.max(Dist)
    rebased_Dist = Dist - max_Dist
    expDist = T.exp(-rebased_Dist)
    expDist = T.fill_diagonal(expDist, 0)
    return expDist / T.sum(expDist, axis=1)  # (M, M) / (M,) : column sum is one


# ----------------------------------------------------------------------------
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
def TrainFn5Member(fnPr, mapping, Marge):
    """

    :param fnPr:
    :param mapping:
    :param Marge:
    :return:
    """

    # define the required symbolic input
    inpl, inpr, inprn = T.ivectors(3)
    lrmapping = T.scalar('lrmapping')  # learning rate

    list_in = [inpl, inpr, inprn, lrmapping]  # highlighting input argument

    Pr = fnPr(mapping.E.T)  # mapping.E ( K x M) matrix
    p = Pr[inpr, inpl]  # (L,)
    pln = Pr[inprn, inpl]
    cost, out = margincost(p, pln, Marge[inpl, inprn])

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
def RankScoreIdx(Pr, idxl, idxr):
    err = []
    for l, r in zip(idxl, idxr):
        err += [np.argsort(np.argsort(
            Pr[:, l])[::-1])[r] + 1]
    return err


# ----------------------------------------------------------------------------
def pca(X=np.array([]), no_dims=30):
    """Runs PCA on the MxN array X in order to reduce its dimensionality to no_dims dimensions."""

    print("Preprocessing the data using PCA...")
    X = X - np.mean(X, 0)  # (M, N) - (N,) using broadcasting
    (_, v) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, v[:, 0:no_dims])
    return Y


# ----------------------------------------------------------------------------
def consine_simi(X=np.array([])):
    """
        Return the cosine similarity matrix for input matrix X
    :param X: (M x N) sample matrix
    :return: P: (M x M) consine similarity measure based on the N features
    """
    inner_product = np.dot(X, X.T)              # (M, M)
    square_magnitude = np.diag(inner_product)    # (M,)
    inv_square_magnitude = 1 / square_magnitude

    inv_square_magnitude[np.isinf(inv_square_magnitude)] = 0

    inv_magnitude = np.sqrt(inv_square_magnitude)
    cosine = inner_product * inv_magnitude
    return cosine.T * inv_magnitude

# ----------------------------------------------------------------------------
def SGDexp(state):
    _log.info(state)
    np.random.seed(state.seed)

    # split the data into training/testing set
    state.ntrain = math.floor(.9 * state.nlinks)
    state.ntest = state.nlinks - state.ntrain
    indices = np.random.permutation(state.nlinks)
    state.trIdxl = state.Idxl[indices[: state.ntrain]]
    state.trIdxr = state.Idxr[indices[: state.ntrain]]

    state.teIdxl = state.Idxl[indices[state.ntrain :]]
    state.teIdxr = state.Idxr[indices[state.ntrain :]]

    state.train = np.mean(RankScoreIdx(simi_X, state.trIdxl, state.trIdxr))
    _log.debug('Content Only: Training set Mean Rank: %s ' % (state.train,))
    state.test = np.mean(RankScoreIdx(simi_X, state.teIdxl, state.teIdxr))
    _log.debug('Content Only: Testing set Mean Rank: %s ' % (state.test,))

    # initialize
    mapping = Mappings(np.random, state.nsamples, state.outdim)  # K x M

    # Function compilation
    apply_fn = eval(state.applyfn)
    trainfunc = TrainFn5Member(apply_fn, mapping, P)

    out = []
    outb = []
    outc = []
    batchsize = math.floor(state.ntrain / state.nbatches)
    state.bestout = np.inf

    _log.info('BEGIN TRAINING')
    timeref = time.time()
    for epoch_count in range(1, state.totepochs + 1):
        # Shuffling
        order = np.random.permutation(state.ntrain)
        trainIdxl = state.trIdxl[order]
        trainIdxr = state.trIdxr[order]

        listidx = np.arange(state.nsamples, dtype='int32')
        listidx = listidx[np.random.permutation(len(listidx))]
        trainIdxrn = listidx[np.arange(state.ntrain) % len(listidx)]


        for _ in range(20):
            for ii in range(state.nbatches):
                tmpl = trainIdxl[ii * batchsize: (ii + 1) * batchsize]
                tmpr = trainIdxr[ii * batchsize: (ii + 1) * batchsize]
                tmprn = trainIdxrn[ii * batchsize: (ii + 1) * batchsize]
                outtmp = trainfunc(tmpl, tmpr, tmprn, state.lrmapping)
                out += [outtmp[0]]
                outb += [outtmp[1]]
                outc += [outtmp[2]]
                # mapping.normalize()

            if np.mean(out) <= state.bestout:
                state.bestout = np.mean(out)
                state.lrmapping *= 1.1
            else:
                state.lrmapping *= .01

        if (epoch_count % state.neval) == 0:
            _log.info('-- EPOCH %s (%s seconds per epoch):' % (epoch_count, (time.time() - timeref) / state.neval))
            _log.info('Cost mean: %s +/- %s      updates: %s%% ' % (np.mean(out), np.std(out), np.mean(outb) * 100))
            _log.debug('Learning rate: %s LeaveOneOut: %s' % (state.lrmapping, np.mean(outc)))

            timeref = time.time()
            Pr = apply_fn(mapping.E.T).eval()
            state.train = np.mean(RankScoreIdx(Pr, state.trIdxl, state.trIdxr))
            _log.debug('Training set Mean Rank: %s  Score: %s' % (state.train, np.mean(Pr[state.trIdxr, state.trIdxl])))
            state.test = np.mean(RankScoreIdx(Pr, state.teIdxl, state.teIdxr))
            _log.debug('Testing set Mean Rank: %s ' % (state.test, ))
            state.cepoch = epoch_count
            f = open(state.savepath + '/' + 'model' + '.pkl', 'wb')  # + str(state.cepoch)
            pickle.dump(mapping, f, -1)
            f.close()
            savemat('pred_dim' + str(state.outdim) + '_method' + state.applyfn +
                    '_maxmarge' + str(state.max_marge) + '.mat', {'mappedX': mapping.E.eval()})
            _log.debug('The saving took %s seconds' % (time.time() - timeref))
            timeref = time.time()

        outb = []
        outc = []
        out = []
        state.bestout = np.inf
        state.lrmapping = 1000.
        f = open(state.savepath + '/' + 'state.pkl', 'wb')
        pickle.dump(state, f, -1)
        f.close()


if __name__ == '__main__':
    _log.info('Start')
    state = DD()

    # check the datapath
    datapath = '../Data/'
    assert datapath is not None

    if 'pickled_data' not in os.listdir('../'):
        os.mkdir('../pickled_data')
    state.savepath = '../pickled_data'

    # load the matlab data file
    mat = loadmat(datapath + dataname + '.mat')
    X = np.array(mat['X'], np.float32)
    I = np.array(mat['I'], np.float32)
    state.Idxl = np.asarray(I[:, 0].flatten() - 1, dtype='int32')  # numpy indexes start from 0
    state.Idxr = np.asarray(I[:, 1].flatten() - 1, dtype='int32')

    state.seed = 213
    state.totepochs = 1200
    state.lrmapping = 10000.
    state.regterm = .0
    state.nsamples, state.nfeatures = np.shape(X)
    state.nlinks = np.shape(state.Idxl)[0]
    state.outdim = outdim
    state.applyfn = applyfn
    state.marge = 2e-3
    state.max_marge = 0.1
    state.nbatches = 1  # mini-batch SGD is not helping here
    state.neval = 10
    state.perplexity = 20.
    state.initial_dim = 300

    Y = pca(X, state.initial_dim)
    # Compute P-values
    Q = x2p(X, 1e-5, state.perplexity)
    #P = P + np.transpose(P)        we don't need to approximate the joint probability as in tsne
    #Q = Q / np.sum(Q)
    Q = np.maximum(Q, 1e-12)
    np.fill_diagonal(Q, 0)
    _log.info('Maximum probability value of the fixed perplexitied distribution: %s' % (np.max(Q, axis=None),))
    dist_X = np.max(Q, axis=None) - Q
    P = T.as_tensor_variable(np.asarray(state.max_marge * dist_X, dtype=theano.config.floatX))

    simi_X = consine_simi(X)
    np.fill_diagonal(simi_X, 0)

    SGDexp(state)
