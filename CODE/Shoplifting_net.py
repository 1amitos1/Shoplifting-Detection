import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='error', category=FutureWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam, SGD
from datetime import date,datetime
#from datetime import datetime
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from keras.models import load_model
import tensorflow as tf
from keras.models import Input, Model
from keras.models import model_from_json
#from keras.optimizers import SGD, Adam
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, Multiply,Add,Concatenate
from keras.layers.core import Lambda
from keras.optimizers import SGD, Adam
import cv2
import numpy as np
import os
#from moviepy.editor import *

import warnings

from termcolor import colored

warnings.filterwarnings("ignore")

class ShopliftingNet:

    def __init__(self,weights_path):
        self.weights_path =weights_path

    def get_rgb(self, input_x):
        rgb = input_x[..., :3]
        return rgb

    # extract the optical flows
    def get_opt(self, input_x):
        opt = input_x[..., 3:5]
        return opt

    # extract slow fast input
    def data_layer(self, input, stride):
        return tf.gather(input, tf.range(0, 64, stride), axis=1)


    def sample(self, input, stride):
        return tf.gather(input, tf.range(0, input.shape[1], stride), axis=1)


    def temporalPooling(self, fast_opt, fast_rgb):
        fast_temoral_poll = Multiply()([fast_rgb, fast_opt])
        fast_temoral_poll = MaxPooling3D(pool_size=(8, 1, 1))(fast_temoral_poll)
        return fast_temoral_poll


    def merging_block(self, x):
        x = Conv3D(
            64, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(x)

        x = Conv3D(
            64, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(x)

        x = MaxPooling3D(pool_size=(2, 2, 2))(x)

        x = Conv3D(
            64, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(x)

        x = Conv3D(
            64, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(x)

        x = MaxPooling3D(pool_size=(2, 2, 2))(x)

        x = Conv3D(
            128, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(x)

        x = Conv3D(
            128, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(x)

        x = MaxPooling3D(pool_size=(2, 3, 3))(x)
        return x


    def get_Flow_gate_fast_path(self, fast_input):
        inputs = fast_input

        connection_dic = {}

        rgb = Lambda(self.get_rgb, output_shape=None)(inputs)

        ##################################################### RGB channel
        # 1
        rgb = Conv3D(
            16, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(rgb)

        rgb = Conv3D(
            16, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(rgb)

        rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)

        rgb = Conv3D(
            16, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(rgb)

        rgb = Conv3D(
            16, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(rgb)

        rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)
        # con1
        # print(f"fast con_1-{rgb.shape}")
        lateral = Lambda(self.sample, arguments={'stride': 18}, name="con_1")(rgb)
        connection_dic.update({"con-1": lateral})

        # 2

        rgb = Conv3D(
            32, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(rgb)

        rgb = Conv3D(
            32, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(rgb)

        rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)

        rgb = Conv3D(
            32, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(rgb)
        rgb = Conv3D(
            32, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(rgb)

        rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)

        # print(f"fast con_2-{rgb.shape}")
        # connection_dic.update({"con-2": rgb})
        lateral = Lambda(self.sample, arguments={'stride': 18}, name="con_2")(rgb)
        connection_dic.update({"con-2": lateral})

        # 3

        return rgb,  connection_dic


    def get_Flow_gate_slow_path(self, slow_input, connection_dic):
        # inputs = Input(shape=(64, 224, 224, 5))
        inputs = slow_input
        rgb = Lambda(self.get_rgb, output_shape=None)(inputs)
        #print(f"len ={connection_dic.items()}")
        con_1 = connection_dic.get('con-1')
        con_2 = connection_dic.get('con-2')

        ##################################################### RGB channel
        rgb = Conv3D(
            16, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(rgb)

        rgb = Conv3D(
            16, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(rgb)

        rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)

        rgb = Conv3D(
            16, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(rgb)

        rgb = Conv3D(
            16, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(rgb)

        rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)

        # con1
        # print(f"slow con_1-{rgb.shape}")
        # print(f"con-1 from fast {connection_dic.get('con-1')}")

        ans1 = Add(name="connection_1_rgb")([rgb, con_1])

        rgb = Conv3D(
            32, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(ans1)

        rgb = Conv3D(
            32, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(rgb)

        rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)

        rgb = Conv3D(
            32, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(rgb)
        rgb = Conv3D(
            32, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(rgb)

        rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)

        # con2
        # print(f"slow con_2-{rgb.shape}")
        # print(f"con-2 from fast {connection_dic.get('con-2').shape}")
        # con_2 = connection_dic.get('con-1')
        ans2 = Add(name="connection_2_rgb")([rgb, con_2])
        # ans = Add([rgb,con_1])
        # print(f"[2]rgb -{rgb.shape}")
        # print(f"[2]Add -{ans2.shape}")
        # ans2 = Multiply()([rgb, con_2])
        # print(f"[2]Multiply -{ans2.shape}")
        # rgb = ans2
        #
        # print(f"hereeeeee[-][-][-]{rgb.shape}")
        # return rgb
        x = ans2
        x = MaxPooling3D(pool_size=(1, 2, 2))(x)
        # print(x.shape)
        # x = MaxPooling3D(pool_size=(8, 1, 1))(x)

        # x=ans2
        ##################################################### Merging Block
        x = Conv3D(
            64, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(x)

        x = Conv3D(
            64, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(x)

        x = MaxPooling3D(pool_size=(2, 2, 2))(x)

        x = Conv3D(
            64, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(x)

        x = Conv3D(
            64, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(x)

        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        # print(f"hereeeeeee {x.shape}")

        x = Conv3D(
            128, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(x)

        x = Conv3D(
            128, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
            padding='same')(x)



        return x


    def gate_flow_slow_fast_network_builder(self):
        clip_shape = [64, 224, 224, 3]
        tau = 16
        clip_input = Input(shape=clip_shape)

        slow_input = Lambda(self.data_layer, arguments={'stride': tau}, name='slow_input')(clip_input)
        # print(slow_input.shape)

        # fast_input = Lambda(data_layer, arguments={'stride': int(tau / alpha)}, name='fast_input')(clip_input)
        fast_input = clip_input

        # build fast path networks
        fast_rgb,  connection = self.get_Flow_gate_fast_path(fast_input)

        # get slow network

        slow_rgb = self.get_Flow_gate_slow_path(slow_input, connection)

       # print(f"[1][+][+] here\nslow_rgb {slow_rgb.shape}\nfast_rgb {fast_rgb.shape}\n")
        # temporal Pooling
        #fast_res_temporal_Pooling = self.temporalPooling(fast_opt, fast_rgb)
        # print(f"res-temporalPooling {fast_res_temporal_Pooling.shape}")

        # merging block
        merging_block_fast_res = self.merging_block(fast_rgb)
        #print(f"[2] merging_block_fast_res = {merging_block_fast_res.shape}")
        #merging_block_fast_res = self.merging_block(fast_res_temporal_Pooling)
        # print("Exit")

        # print(merging_block_fast_res.shape)
        # print(f"slow_rgb-{slow_rgb.shape}")
        # conact slow_rgb with merging_block_fast_res
        x = Add(name="ADD_slow_rgb_ans_fast_rgb_opt")([merging_block_fast_res, slow_rgb])

        #print(f"[3] model = {x.shape}")
        ##########FC layer#########################################
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        # x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)

        # Build the model
        pred = Dense(3, activation='softmax')(x)
        model = Model(inputs=clip_input, outputs=pred)
        return model


    # build model
    def get_gate_flow_slow_fast_model(self):
        """
        build gate_flow_slow_fast without weight_steals
        :return: gate_flow_slow_fast model
        """
        model = self.gate_flow_slow_fast_network_builder()
        model.load_weights(self.weights_path)
        return model

    def load_model_and_weight(self):
        model = self.gate_flow_slow_fast_network_builder()
        model.load_weights(self.weights_path)
        return model

#
# S_net = ShopliftingNet()
# #S_net.get_gate_flow_slow_fast_model()