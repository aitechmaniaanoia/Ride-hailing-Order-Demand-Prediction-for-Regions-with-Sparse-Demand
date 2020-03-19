import joint_model
import utils
import numpy as np
import sys




cnn_f = np.load('Ximage_test.npz')
cnns = cnn_f['arr_0']
cnn_f.close()
out_f = np.load('Ylabel_test.npz')
outs = out_f['arr_0']
out_f.close()
lstms = []
for i in range(1):
	lstm_f = np.load("lstm_%d_test.npz"%(i))
	lstm_chunk = lstm_f['arr_0']
	lstm_f.close()
	lstms.append(lstm_chunk)
lstms=np.concatenate(lstms)
topo_f = np.load("topo.npz")
topos = topo_f['arr_0']
topo_f.close()
topos = np.repeat(topos, (1+cnns.shape[0])/topos.shape[0], axis = 0)[:-1]



#nonzero_examples = [i for i in range(lstms.shape[0]-1) if np.sum(cnns[i,:,1,1]) > 3 ]
#zero_examples = [i for i in range(lstms.shape[0]-1) if not np.sum(cnns[i,:,1,1]) > 3 ]

#recent_only = False
#if recent_only:
  #lstms = np.take(lstms, nonzero_examples, axis=0)
  #cnns = np.take(cnns, nonzero_examples, axis=0)
  #outs = np.take(outs, nonzero_examples, axis=0)
  #topos = np.take(topos, nonzero_examples, axis=0)
#print(len(nonzero_examples))
print(cnns.shape)
print(outs.shape)
print(lstms.shape)
print(topos.shape)

#print(len(nonzero_examples))
#print(len(zero_examples))


N_TEST_EXAMPLES = lstms.shape[0]-1
#N_TEST_EXAMPLES = 100
lstms = lstms[:N_TEST_EXAMPLES]
cnns = cnns[:N_TEST_EXAMPLES]
outs  = outs[:N_TEST_EXAMPLES]
topos = topos[:N_TEST_EXAMPLES]

demand_Z = max([np.max(cnns),np.max(outs)])
cnns = cnns / demand_Z
outs = outs / demand_Z

trained_model = joint_model.build_model(
        lstms,outs,
        None,None,
        cnns,None,
        topos,None,
        joint_model.feature_len,
        weights_path=sys.argv[1],
        #'weights_best.h5'
        do_normalize=True
        )

#eval_ = trained_model.evaluate([cnns, lstms, topos], outs, verbose=2)
#print(eval_)
#print(trained_model.metrics_names)

pred = trained_model.predict([cnns, lstms, topos], verbose=2)
denormalized = pred[:,0] * (pred[:,1] > 0.5) # mask using classifier labels
denormalized *= demand_Z
# for hiero models: make some predictions zero
#np.put(denormalized, zero_examples, 0)
outs *= demand_Z
outs = outs[:,0]
print(denormalized[:10])

# MAPE
def MAE(pred, true):
    return np.mean(np.abs(pred - true))
# MSLE
def MSLE(pred, true):
    return np.mean(np.square(np.log(1+true) - np.log(1+pred)))
# RMSE
def RMSE(pred, true):
    return np.sqrt(np.mean(np.square(pred - true)))
metrics = [MAE,MSLE,RMSE]
for metric in metrics:
    print( metric.__name__, metric( denormalized, outs ) )

#np.savez_compressed("pred_base_log", denormalized)
