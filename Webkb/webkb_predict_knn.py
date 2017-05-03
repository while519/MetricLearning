#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import *
from scipy.io.matlab import loadmat
from scipy.io.matlab import savemat
import logging
import math
import time
import pickle

# experimental parameters
dataname = 'webkb'
applyfn = 'softmax'

# adjustable parameters
outdim = 20
knn = 0
marge_ratio = 10.

FORMAT = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
_log = logging.getLogger(dataname +' experiment')
_log.setLevel(logging.DEBUG)
ch_file = logging.FileHandler(filename= 'predict_' + applyfn + str(outdim) + '_knn' +
                                        str(knn) + '_margeratio' + str(marge_ratio) +  '.log', mode='w')
ch_file.setLevel(logging.DEBUG)
ch_file.setFormatter(FORMAT)
ch = logging.StreamHandler()
ch.setFormatter(FORMAT)
ch.setLevel(logging.DEBUG)
_log.addHandler(ch)
_log.addHandler(ch_file)


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
                    '_marge' + str(state.marge) + '.mat', {'mappedX': mapping.E.eval()})
            _log.debug('The saving took %s seconds' % (time.time() - timeref))
            timeref = time.time()

        outb = []
        outc = []
        out = []
        state.bestout = np.inf
        if state.lrmapping < state.baselr:      # if the learning rate is not growing
            state.baselr *= 0.1
        state.lrmapping = state.baselr
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
    state.lrmapping = 1000.
    state.baselr = state.lrmapping
    state.regterm = .0
    state.nsamples, state.nfeatures = np.shape(X)
    state.nlinks = np.shape(state.Idxl)[0]
    state.outdim = outdim
    state.applyfn = applyfn
    state.marge = marge_ratio / state.nsamples
    state.nbatches = 1  # mini-batch SGD is not helping here
    state.neval = 10
    state.initial_dim = 300


    # cosine similarity measure
    simi_X = consine_simi(X)
    np.fill_diagonal(simi_X, 0)

    # construct the marginal matrix
    #np.sort(simi_X, axis=1)
    dist_X = np.ones(shape=(state.nsamples, state.nsamples))
    knnIdx = np.argsort(-simi_X.copy(), axis=1)[:,:knn]
    for rowind in range(state.nsamples):
        dist_X[rowind, knnIdx[rowind,:]] = .0

    P = T.as_tensor_variable(np.asarray(state.marge * dist_X, dtype=theano.config.floatX))

    # start the experiments
    SGDexp(state)
