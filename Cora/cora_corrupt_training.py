#!/usr/bin/env python

from model import *

FORMAT = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
_log = logging.getLogger('Cora experiment')
_log.setLevel(logging.DEBUG)
ch_file = logging.FileHandler(filename='my.log', mode='w')
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

    # initialize
    mapping = Mappings1(np.random, state.nfeatures, state.outdim)  # K x N

    # Function compilation
    apply_fn = eval(state.applyfn)
    trainfunc = TrainFnMember(apply_fn, X, mapping, marge=state.marge)

    out = []
    outb = []
    outc = []
    batchsize = math.floor(state.nlinks / state.nbatches)
    state.bestout = np.inf

    _log.info('BEGIN TRAINING')
    timeref = time.time()
    order = np.random.permutation(state.nlinks)
    for epoch_count in range(1, state.totepochs + 1):
        # Shuffling
        trainIdxl = state.Idxl[order]
        trainIdxr = state.Idxr[order]

        listidx = np.arange(state.nsamples, dtype='int32')
        listidx = listidx[np.random.permutation(len(listidx))]
        trainIdxrn = listidx[np.arange(state.nlinks) % len(listidx)]

        for ii in range(state.nbatches):
            tmpl = trainIdxl[ii * batchsize: (ii + 1) * batchsize]
            tmpr = trainIdxr[ii * batchsize: (ii + 1) * batchsize]
            tmprn = trainIdxrn[ii * batchsize: (ii + 1) * batchsize]
            outtmp = trainfunc(tmpl, tmpr, tmprn, state.lrmapping)
            out += [outtmp[0]]
            outb += [outtmp[1]]
            outc += [outtmp[2]]

        if outtmp[0] < state.bestout:
            # state.bestout = np.mean(out)
            state.lrmapping *= 1.01
        else:
            state.lrmapping *= 0.4
            # if state.lrmapping < np.exp(-30):
            #     state.lrmapping = 200

        if (epoch_count % state.neval) == 0:
            _log.info('-- EPOCH %s (%s seconds per epoch):' % (epoch_count, (time.time() - timeref) / state.neval))
            _log.info('Cost mean: %s +/- %s      updates: %s%% ' % (np.mean(out), np.std(out), np.mean(outb) * 100))
            _log.debug('Learning rate: %s LeaveOneOut: %s' % (state.lrmapping, np.mean(outc)))
            outb = []
            outc = []

            if np.mean(out) < state.bestout:
                state.bestout = np.mean(out)
                timeref = time.time()
                Pr = apply_fn(X.dot(mapping.E.T)).eval()
                state.train = np.mean(RankScoreIdx(Pr, state.Idxl, state.Idxr))
                _log.debug('Training set Mean Rank: %s  Score: %s' % (state.train, np.mean(Pr[state.Idxr, state.Idxl])))
                state.cepoch = epoch_count
                f = open(state.savepath + '/' + 'model' + '.pkl', 'wb')  # + str(state.cepoch)
                pickle.dump(mapping, f, -1)
                f.close()
                _log.debug('The saving took %s seconds' % (time.time() - timeref))
            timeref = time.time()
            out = []

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
    mat = loadmat(datapath + 'processed_cora.mat')
    trY = np.array(mat['trY'], np.float32)
    trIdx1 = np.array(mat['trRowIdx'], np.int32)
    trIdx2 = np.array(mat['trColumnIdx'], np.int32)

    state.seed = 213
    state.totepochs = 1000
    state.lrmapping = .001
    state.regterm = .0
    X = T.as_tensor_variable(np.asarray(trY, dtype=theano.config.floatX))  # content matrix
    # Indpairs = T.cast(theano.shared(np.asarray(np.concatenate([trIdx1, trIdx2], axis=1),
    #                                    dtype=theano.config.floatX)), 'int32')      # pairs for citation linkages
    state.Idxl = np.asarray(trIdx1.flatten() - 1, dtype='int32')  # matlab to numpy indexes conversion
    state.Idxr = np.asarray(trIdx2.flatten() - 1, dtype='int32')
    state.nsamples, state.nfeatures = np.shape(trY)
    state.nlinks = np.shape(trIdx1)[0]
    state.outdim = 30
    state.applyfn = 'softcauchy'
    state.marge = 0.005
    state.nbatches = 1000 # mini-batch SGD is not helping here
    state.neval = 1
    state.lrparam = 1.

    SGDexp(state)
