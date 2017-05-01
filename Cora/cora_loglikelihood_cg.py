#!/usr/bin/env python

from model import *
import scipy.optimize

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
	# x0 = np.asarray(np.random.random(state.nfeatures * state.outdim), dtype=theano.config.floatX)
	mapping = Mappings1(np.random, state.nfeatures, state.outdim)
	x0 = np.asarray(mapping.E.flatten().eval(), dtype=theano.config.floatX)

	# Function compilation
	apply_fn = eval(state.applyfn)
	trainfunc, traingrad = TrainFn2Member(apply_fn, X)

	out = []
	batchsize = math.floor(state.nlinks / state.nbatches)
	state.bestout = 0

	_log.info('BEGIN TRAINING')
	timeref = time.time()
	for epoch_count in range(1, state.totepochs + 1):
		# Shuffling
		order = np.random.permutation(state.nlinks)
		trainIdxl = state.Idxl[order]
		trainIdxr = state.Idxr[order]

		for ii in range(state.nbatches):
			outtmp1 = trainfunc(x0, trainIdxl[ii * batchsize: (ii + 1) * batchsize],
								trainIdxr[ii * batchsize: (ii + 1) * batchsize],
								state.outdim)
			outtmp2 = traingrad(x0, trainIdxl[ii * batchsize: (ii + 1) * batchsize],
								trainIdxr[ii * batchsize: (ii + 1) * batchsize],
								state.outdim)

			xopt = scipy.optimize.fmin_cg(trainfunc, x0, fprime=traingrad,
										  args=(trainIdxl[ii * batchsize: (ii + 1) * batchsize],
												trainIdxr[ii * batchsize: (ii + 1) * batchsize],
												state.outdim), maxiter=100, disp=1)

			outtmp1 = trainfunc(xopt, trainIdxl[ii * batchsize: (ii + 1) * batchsize],
								trainIdxr[ii * batchsize: (ii + 1) * batchsize],
								state.outdim)
			x0 = xopt
			out += [outtmp1]

		if np.mean(out) < state.bestout:
			state.bestout = np.mean(out)

		if (epoch_count % state.neval) == 0:
			_log.info('-- EPOCH %s (%s seconds per epoch):' % (epoch_count, (time.time() - timeref) / state.neval))
			_log.info('Cost mean: %s +/- %s  Best: %s' % (np.mean(out), np.std(out), state.bestout))

			if np.mean(out) < state.bestout:
				timeref = time.time()
				state.cepoch = epoch_count
				f = open(state.savepath + '/' + 'model' + '.pkl', 'wb')  # + str(state.cepoch)
				pickle.dump(xopt, f, -1)
				f.close()
				_log.info('The saving took %s seconds' % (time.time() - timeref))
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
	state.totepochs = 30
	state.lrmapping = 1.
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
	state.marge = 0.2
	state.nbatches = 1  # mini-batch SGD is not helping here
	state.neval = 1

	SGDexp(state)
