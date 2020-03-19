import tensorflow as tf
from utils import *
from keras import backend as K
import numpy as np
import pandas as pd
import argparse
from keras.models import Sequential, Model
from keras import activations
from keras.engine.topology import Layer, InputSpec
from keras.utils import conv_utils
from keras.layers import LSTM, InputLayer, Dense, Input, Flatten, concatenate, Reshape
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
from keras import metrics
from keras.layers.normalization import BatchNormalization
from random import randint
import pickle

batch_size = 2 # set in build()
mean_label = 0.5 # set in build()
label_max = 1 # set in build()
label_min = 0 # set in build()


max_epoch = 100
seq_len = 6
hidden_dim = 128
threshold = 10.0
loss_lambda = 10000.0
feature_len = 11 # number of lstm features
local_image_size = 3 # size of neighborhood for CNN images
cnn_hidden_dim_first = 8
toponet_len = 32

#num_feature = 100
#maxtruey = 1
#mintruey = 0
#eps = 1e-5
#len_valid_id = 0

sess = tf.Session()
K.set_session(sess)


class Local_Seq_Conv(Layer):

    def __init__(self, output_dim, seq_len, feature_size, kernel_size, activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', padding='same', strides=(1, 1), **kwargs):
        super(Local_Seq_Conv, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.bias_initializer = bias_initializer
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.padding = padding
        self.strides = strides
        self.activation = activations.get(activation)

    def build(self, input_shape):
        #batch_size = input_shape[0]
        batch_size = 2
        self.kernel = []
        self.bias = []
        for eachlen in range(self.seq_len):
            self.kernel += [self.add_weight(shape=self.kernel_size,
                                            initializer=self.kernel_initializer,
                                            trainable=True, name='kernel_{0}'.format(eachlen))]

            self.bias += [self.add_weight(shape=(self.kernel_size[-1],),
                                          initializer=self.bias_initializer,
                                          trainable=True, name='bias_{0}'.format(eachlen))]
        self.build = True

    def call(self, inputs):
        output = []
        for eachlen in range(self.seq_len):

            tmp = K.bias_add(K.conv2d(inputs[:, eachlen, :, :, :], self.kernel[eachlen],
                                      strides=self.strides, padding=self.padding), self.bias[eachlen])

            if self.activation is not None:
                output += [self.activation(tmp)]

        output = tf.stack(output, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3], self.output_dim)


def build_model(trainX, trainY, testX, testY, trainimage, testimage, traintopo, testtopo, feature_len,
    weights_path=None, do_normalize = False, class_weights=[1,1] ):
    # X_train, Y_train, X_test, Y_test = Featureset_get()

    try:
      mean_label = np.mean(trainY)
      label_max = np.max(trainY)
      label_min = np.min(trainY)
    except TypeError:
      pass

    trainClass = np.minimum( np.ones_like(trainY), trainY )
    if do_normalize:
        print("normalizing...")
        demand_Z = max([np.max(trainimage),np.max(trainY)])
        trainimage = trainimage / demand_Z
        trainY     = trainY     / demand_Z
    trainY = np.concatenate( [trainY, trainClass], axis=1 )

    image_input = Input(shape=(seq_len, local_image_size,
                               local_image_size, None), name='cnn_input')
    spatial = Local_Seq_Conv(output_dim=cnn_hidden_dim_first, seq_len=seq_len, feature_size=feature_len,
                             kernel_size=(3, 3, 1, cnn_hidden_dim_first), activation='relu',
                             kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same',
                             strides=(1, 1))(image_input)
    spatial = BatchNormalization()(spatial)
    # spatial = Local_Seq_Pooling(seq_len=seq_len)(spatial)
    spatial = Local_Seq_Conv(output_dim=cnn_hidden_dim_first, seq_len=seq_len, feature_size=feature_len,
                             kernel_size=(3, 3, cnn_hidden_dim_first, cnn_hidden_dim_first), activation='relu',
                             kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same',
                             strides=(1, 1))(spatial)
    spatial = BatchNormalization()(spatial)
    # spatial = Local_Seq_Pooling(seq_len=seq_len)(spatial)
    spatial = Local_Seq_Conv(output_dim=cnn_hidden_dim_first, seq_len=seq_len, feature_size=feature_len,
                             kernel_size=(3, 3, cnn_hidden_dim_first, cnn_hidden_dim_first), activation='relu',
                             kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same',
                             strides=(1, 1))(spatial)
    # spatial = Local_Seq_Pooling(seq_len=seq_len)(spatial)
    # spatial = BatchNormalization()(spatial)
    # spatial = Local_Seq_Conv(output_dim=cnn_hidden_dim_first, seq_len=seq_len, feature_size=feature_len, kernel_size=(3, 3, cnn_hidden_dim_first, cnn_hidden_dim_first), activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same', strides=(1, 1))(spatial)
    # spatial = BatchNormalization()(spatial)
    # spatial = Local_Seq_Pooling(seq_len=seq_len)(spatial)
    # spatial = Local_Seq_Conv(output_dim=cnn_hidden_dim_first, seq_len=seq_len, feature_size=feature_len, kernel_size=(3, 3, cnn_hidden_dim_first, cnn_hidden_dim_first), activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same', strides=(1, 1))(spatial)
    spatial = Flatten()(spatial)
    spatial = Reshape(target_shape=(seq_len, -1))(spatial)
    spatial_out = Dense(units=64, activation='relu')(spatial)

    lstm_input = Input(shape=(seq_len, feature_len),
                       dtype='float32', name='lstm_input')

    x = concatenate([lstm_input, spatial_out], axis=-1)
    # lstm_out = Dense(units=128, activation=relu)(x)
    lstm_out = LSTM(units=hidden_dim, return_sequences=False, dropout=0)(x)

    topo_input = Input(shape=(toponet_len,), dtype='float32', name='topo_input')
    topo_emb = Dense(units=6, activation='tanh')(topo_input)
    static_dynamic_concate = concatenate([lstm_out, topo_emb], axis=-1)

    res = Dense(units=2, activation='sigmoid')(static_dynamic_concate)
    model = Model(inputs=[image_input, lstm_input, topo_input],
                  outputs=res)
    sgd = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    #model.compile(loss=mean_absolute_percentage_error_revise, optimizer=sgd,
                  #metrics=[metrics.mae])
    model.compile(loss=joint_error( class_weights ), optimizer=sgd,
                  metrics=[metrics.mae])
    earlyStopping = EarlyStopping(
        monitor='val_loss', patience=10, verbose=0, mode='min')
    mc = ModelCheckpoint('weights_best.h5', 
                                     save_weights_only=True, monitor='val_loss',save_best_only=True)
    csv_logger = CSVLogger('training.log')
    if weights_path is None:
      model.fit([trainimage, trainX, traintopo], trainY, batch_size=batch_size, epochs=max_epoch, validation_split=0.1,
              callbacks=[earlyStopping,mc,csv_logger])
      model.save('weights_fully_trained.h5')
    else:
      model.load_weights(weights_path)
    return model
