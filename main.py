#import weather
import model
import utils
import numpy as np

#cnn_f = np.load("cnn_0.npz")
#cnns = cnn_f['arr_0']
#outs = cnn_f['arr_1']
#cnn_f.close()
#cnns = cnns.reshape((-1,12,3,3,1))
cnn_f = np.load('Ximage.npz')
cnns = cnn_f['arr_0']
cnn_f.close()
out_f = np.load('Ylabel.npz')
outs = out_f['arr_0']
out_f.close()
lstms = []
for i in range(1):
	lstm_f = np.load("lstm_%d.npz"%(i))
	lstm_chunk = lstm_f['arr_0']
	lstm_f.close()
	lstms.append(lstm_chunk)
lstms=np.concatenate(lstms)
topo_f = np.load("topo.npz")
topos = topo_f['arr_0']
topo_f.close()
topos = np.repeat(topos, (1+cnns.shape[0])/topos.shape[0], axis = 0)[:-1]

####################
# Trying to deal with imbalance:
#
# find all training examples where there is *some* nonzero demand somewhere in the 
# CNN data:
nonzero_examples = [i for i in range(lstms.shape[0]) if np.sum(cnns[i,:,1,1]) > 3 ]
# OR
# find all training examples where prediction is nonzero:
#nonzero_examples = [i for i in range(lstms.shape[0]) if outs[i] > 0]

lstms = np.take(lstms, nonzero_examples, axis=0)
cnns = np.take(cnns, nonzero_examples, axis=0)
outs = np.take(outs, nonzero_examples, axis=0)
topos = np.take(topos, nonzero_examples, axis=0)

N_TRAIN_EXAMPLES = lstms.shape[0]-1
#N_TRAIN_EXAMPLES = 10000
N_TEST_EXAMPLES=0
lstms = lstms[:N_TRAIN_EXAMPLES]
cnns = cnns[:N_TRAIN_EXAMPLES]
outs  = outs[:N_TRAIN_EXAMPLES]
topos = topos[:N_TRAIN_EXAMPLES]
#print(len(nonzero_examples))
print(cnns.shape)
print(outs.shape)
print(lstms.shape)
print(topos.shape)


# LSTM inputs
# "Context features" at each time step: quote from paper:
# average demand value in the last four timeintervals, spatial features (e.g., longitude and latitude of the region center), meteorological features (e.g., weather condition), event features (e.g., holiday)
#trainX = np.ones( (N_TRAIN_EXAMPLES, model.seq_len, model.feature_len) )
trainX = lstms
testX = np.ones( (N_TEST_EXAMPLES, model.seq_len, model.feature_len) )

# Predictions: demand in each region
# Should probably have shape (N_TRAIN_EXAMPLES, 1) if
# we are just predicting demand. More dimensions = more
# predicted features eg avg number of riders, avg cost,
# etc.
#trainY = np.ones( (N_TRAIN_EXAMPLES, model.feature_len) )
trainY = outs
testY  = np.ones( (N_TEST_EXAMPLES, model.feature_len) )

# Training images: SxSx1 heatmap for
# each region for each timestep
#trainimg = np.ones( (N_TRAIN_EXAMPLES, model.seq_len, model.local_image_size, model.local_image_size, 1) )
trainimg = cnns
testimg = None
# Figure 1 shows the toponet embedding as part of the network,
# but the code looks like it expects a pretrained topology embedding.
# We should update the code to add a topology network embedding
# layer and train this layer jointly with everything else.
#traintopo = np.ones( (N_TRAIN_EXAMPLES, model.toponet_len,) )
traintopo = topos
testtopo  = None

model.build_model(trainX,trainY,
        testX, testY,
        trainimg, testimg,
        traintopo, testtopo,
        model.feature_len,
        do_normalize = True
        )

#errors - model has no attribute fit
#model.fit()

# trainimg = image_input.shape = (seq_len, local_image_size, local_image_size, None)
# traintopo = topo_input.shape = (toponet_len,)
# testtopo = not used
# trainX = lstm_input = (seq_len, feature_len)
