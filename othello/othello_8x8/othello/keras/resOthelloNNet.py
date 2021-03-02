import sys
sys.path.append('..')
from utils import *

import argparse
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

class OthelloNNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        

         # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y

        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        
        resnet_v12 = self.resnet_v1(inputs = x_image, num_res_blocks = 2)
        gap1 = GlobalAveragePooling2D()(resnet_v12)
        #res_flatten = Flatten()(resnet_v16)       
        #s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(res_flatten))))  # batch_size x 1024
        #s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))          # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(gap1)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(gap1)                    # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))
        
        
    def resnet_layer(self, inputs, num_filter = 16, kernel_size = 3, strides = 1, activation = 'relu', batch_normalization = True, conv_first = True, padding = 'same'):
        
        conv = Conv2D(num_filter, 
                    kernel_size = kernel_size, 
                    strides = strides, 
                    padding = padding,
                    use_bias = False,  
                    kernel_regularizer = l2(1e-4))
        
        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization(axis=3)(x)
            if activation is not None:
                x = Activation(activation)(x)
            
        else:
            if batch_normalization:
                x = BatchNormalization(axis=3)(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x

    
        

    def resnet_v1(self, inputs, num_res_blocks):
        x = inputs
        for  i in range(1):
            resnet = self.resnet_layer(inputs = x, num_filter = 128)    
            resnet = self.resnet_layer(inputs = resnet, num_filter = 128, activation = None)     
            resnet = add([resnet, x])
            resnet = Activation('relu')(resnet)
            x = resnet

        for i in range(2):
            if(i == 0):
                resnet = self.resnet_layer(inputs = x, num_filter = 256,strides=2)
                resnet = self.resnet_layer(inputs = resnet, num_filter = 256, activation = None)
            else:
                resnet = self.resnet_layer(inputs = x, num_filter = 256)
                resnet = self.resnet_layer(inputs = resnet, num_filter = 256, activation = None)
            if(i == 0):
                x = self.resnet_layer(inputs = x, num_filter = 256, strides=2)
            resnet = add([resnet, x])
            resnet = Activation('relu')(resnet)
            x = resnet

        for i in range(2):
            if(i == 0):
                resnet = self.resnet_layer(inputs = x, num_filter = 512,strides=2)
                resnet = self.resnet_layer(inputs = resnet, num_filter = 512, activation = None)
            else:
                resnet = self.resnet_layer(inputs = x, num_filter = 512)
                resnet = self.resnet_layer(inputs = resnet, num_filter = 512, activation = None)
            if(i == 0):
                x = self.resnet_layer(inputs = x, num_filter = 512, strides=2)
            resnet = add([resnet, x])
            resnet = Activation('relu')(resnet)
            x = resnet
        
        

        return x

        

        