import warnings

from Shoplifting.Shoplifting_net import ShopliftingNet

warnings.filterwarnings("ignore")
warnings.simplefilter(action='error', category=FutureWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
from keras.models import load_model
#from keras.optimizers import Adam, SGD
from datetime import date,datetime
#from datetime import datetime
import tensorflow as tf
from keras.models import Input, Model
from keras.models import model_from_json
#from keras.optimizers import SGD, Adam
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, Multiply,Add,Concatenate
from keras.layers.core import Lambda
import cv2
import numpy as np
import os
#from moviepy.editor import *

import warnings

from termcolor import colored

warnings.filterwarnings("ignore")

class Shoplifting_Live():

    def __init__(self):
        #shoplifting_weight_path = r"E:\FINAL_PROJECT_DATA\2021\Shoplifting_detection\Shoplifting\weight_steals\GATE_FLOW_SLOW_FAST_RGB_ONLY\weights_at_epoch_5_rgb_72ACC_THE_BAST.h5"
        shoplifting_weight_path = r"/Shoplifting/weight_steals\GATE_FLOW_SLOW_FAST_RGB_ONLY\weights_at_epoch_8_75___good.h5"
        #r"weights_at_epoch_1_new_train.h5"
        self.weight_path_Shoplifting = shoplifting_weight_path
        self.ShopliftingNet_RGB = ShopliftingNet(shoplifting_weight_path)
        self.shoplifting_model = None
        self.ShopliftingNet_RGB_model = None
        self.frames = None
        self.test_index = 0

        #self.build_abuse_AND_fall_models()
    #
    # def get_rgb(self, input_x):
    #     rgb = input_x[..., :3]
    #     return rgb
    # # extract the optical flows
    # def get_opt(self, input_x):
    #     opt = input_x[..., 3:5]
    #     return opt
    # # extract slow fast input
    # def data_layer(self, input, stride):
    #     return tf.gather(input, tf.range(0, 64, stride), axis=1)
    # def sample(self, input, stride):
    #     return tf.gather(input, tf.range(0, input.shape[1], stride), axis=1)
    # def temporalPooling(self, fast_opt, fast_rgb):
    #     fast_temoral_poll = Multiply()([fast_rgb, fast_opt])
    #     fast_temoral_poll = MaxPooling3D(pool_size=(8, 1, 1))(fast_temoral_poll)
    #     return fast_temoral_poll
    # def merging_block(self, x):
    #     x = Conv3D(
    #         64, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(x)
    #
    #     x = Conv3D(
    #         64, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(x)
    #
    #     x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    #
    #     x = Conv3D(
    #         64, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(x)
    #
    #     x = Conv3D(
    #         64, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(x)
    #
    #     x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    #
    #     x = Conv3D(
    #         128, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(x)
    #
    #     x = Conv3D(
    #         128, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(x)
    #
    #     x = MaxPooling3D(pool_size=(2, 3, 3))(x)
    #     return x
    #
    # def get_Flow_gate_fast_path(self, fast_input):
    #     inputs = fast_input
    #
    #     connection_dic = {}
    #
    #     rgb = Lambda(self.get_rgb, output_shape=None)(inputs)
    #
    #     opt = Lambda(self.get_opt, output_shape=None)(inputs)
    #
    #     ##################################################### RGB channel
    #     # 1
    #     rgb = Conv3D(
    #         16, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(rgb)
    #
    #     rgb = Conv3D(
    #         16, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(rgb)
    #
    #     rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)
    #
    #     rgb = Conv3D(
    #         16, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(rgb)
    #
    #     rgb = Conv3D(
    #         16, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(rgb)
    #
    #     rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)
    #     # con1
    #     # print(f"fast con_1-{rgb.shape}")
    #     lateral = Lambda(self.sample, arguments={'stride': 18}, name="con_1")(rgb)
    #     connection_dic.update({"con-1": lateral})
    #
    #     # 2
    #
    #     rgb = Conv3D(
    #         32, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(rgb)
    #
    #     rgb = Conv3D(
    #         32, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(rgb)
    #
    #     rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)
    #
    #     rgb = Conv3D(
    #         32, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(rgb)
    #     rgb = Conv3D(
    #         32, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(rgb)
    #
    #     rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)
    #
    #     # print(f"fast con_2-{rgb.shape}")
    #     # connection_dic.update({"con-2": rgb})
    #     lateral = Lambda(self.sample, arguments={'stride': 18}, name="con_2")(rgb)
    #     connection_dic.update({"con-2": lateral})
    #
    #     # 3
    #
    #     ##################################################### Optical Flow channel
    #     opt = Conv3D(
    #         16, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(opt)
    #
    #     opt = Conv3D(
    #         16, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(opt)
    #
    #     opt = MaxPooling3D(pool_size=(1, 2, 2))(opt)
    #
    #     opt = Conv3D(
    #         16, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(opt)
    #
    #     opt = Conv3D(
    #         16, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(opt)
    #
    #     opt = MaxPooling3D(pool_size=(1, 2, 2))(opt)
    #
    #     opt = Conv3D(
    #         32, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(opt)
    #
    #     opt = Conv3D(
    #         32, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(opt)
    #
    #     opt = MaxPooling3D(pool_size=(1, 2, 2))(opt)
    #
    #     opt = Conv3D(
    #         32, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='sigmoid',
    #         padding='same')(opt)
    #
    #     opt = Conv3D(
    #         32, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='sigmoid',
    #         padding='same')(opt)
    #
    #     opt = MaxPooling3D(pool_size=(1, 2, 2))(opt)
    #
    #     return rgb, opt, connection_dic
    # def get_Flow_gate_slow_path(self, slow_input, connection_dic):
    #     # inputs = Input(shape=(64, 224, 224, 5))
    #     inputs = slow_input
    #     rgb = Lambda(self.get_rgb, output_shape=None)(inputs)
    #
    #     con_1 = connection_dic.get('con-1')
    #     con_2 = connection_dic.get('con-2')
    #
    #     ##################################################### RGB channel
    #     rgb = Conv3D(
    #         16, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(rgb)
    #
    #     rgb = Conv3D(
    #         16, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(rgb)
    #
    #     rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)
    #
    #     rgb = Conv3D(
    #         16, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(rgb)
    #
    #     rgb = Conv3D(
    #         16, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(rgb)
    #
    #     rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)
    #
    #     # con1
    #     # print(f"slow con_1-{rgb.shape}")
    #     # print(f"con-1 from fast {connection_dic.get('con-1')}")
    #
    #     ans1 = Add(name="connection_1_rgb")([rgb, con_1])
    #
    #     rgb = Conv3D(
    #         32, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(ans1)
    #
    #     rgb = Conv3D(
    #         32, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(rgb)
    #
    #     rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)
    #
    #     rgb = Conv3D(
    #         32, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(rgb)
    #     rgb = Conv3D(
    #         32, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(rgb)
    #
    #     rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)
    #
    #     # con2
    #     # print(f"slow con_2-{rgb.shape}")
    #     # print(f"con-2 from fast {connection_dic.get('con-2').shape}")
    #     # con_2 = connection_dic.get('con-1')
    #     ans2 = Add(name="connection_2_rgb")([rgb, con_2])
    #     # ans = Add([rgb,con_1])
    #     # print(f"[2]rgb -{rgb.shape}")
    #     # print(f"[2]Add -{ans2.shape}")
    #     # ans2 = Multiply()([rgb, con_2])
    #     # print(f"[2]Multiply -{ans2.shape}")
    #     # rgb = ans2
    #     #
    #     # print(f"hereeeeee[-][-][-]{rgb.shape}")
    #     # return rgb
    #     x = ans2
    #     x = MaxPooling3D(pool_size=(1, 2, 2))(x)
    #     # print(x.shape)
    #     # x = MaxPooling3D(pool_size=(8, 1, 1))(x)
    #
    #     # x=ans2
    #     ##################################################### Merging Block
    #     x = Conv3D(
    #         64, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(x)
    #
    #     x = Conv3D(
    #         64, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(x)
    #
    #     x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    #
    #     x = Conv3D(
    #         64, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(x)
    #
    #     x = Conv3D(
    #         64, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(x)
    #
    #     x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    #     # print(f"hereeeeeee {x.shape}")
    #
    #     x = Conv3D(
    #         128, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(x)
    #
    #     x = Conv3D(
    #         128, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu',
    #         padding='same')(x)
    #
    #     # print(x.shape)
    #     # x = MaxPooling3D(pool_size=(2, 3, 3))(x)
    #
    #     # x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    #     # print(f"--final{x.shape}")
    #
    #     return x
    # def gate_flow_slow_fast_network_builder(self):
    #     clip_shape = [64, 224, 224, 5]
    #     tau = 16
    #     clip_input = Input(shape=clip_shape)
    #
    #     slow_input = Lambda(self.data_layer, arguments={'stride': tau}, name='slow_input')(clip_input)
    #     # print(slow_input.shape)
    #
    #     # fast_input = Lambda(data_layer, arguments={'stride': int(tau / alpha)}, name='fast_input')(clip_input)
    #     fast_input = clip_input
    #
    #     # build fast path networks
    #     fast_rgb, fast_opt, connection = self.get_Flow_gate_fast_path(fast_input)
    #
    #     # get slow network
    #
    #     slow_rgb = self.get_Flow_gate_slow_path(slow_input, connection)
    #
    #     # temporal Pooling
    #     fast_res_temporal_Pooling = self.temporalPooling(fast_opt, fast_rgb)
    #     # print(f"res-temporalPooling {fast_res_temporal_Pooling.shape}")
    #
    #     # merging block
    #     merging_block_fast_res = self.merging_block(fast_res_temporal_Pooling)
    #     # print("Exit")
    #
    #     # print(merging_block_fast_res.shape)
    #     # print(f"slow_rgb-{slow_rgb.shape}")
    #     # conact slow_rgb with merging_block_fast_res
    #     x = Add(name="ADD_slow_rgb_ans_fast_rgb_opt")([merging_block_fast_res, slow_rgb])
    #
    #     ##########FC layer#########################################
    #     x = Flatten()(x)
    #     x = Dense(128, activation='relu')(x)
    #     # x = Dropout(0.2)(x)
    #     x = Dense(32, activation='relu')(x)
    #
    #     # Build the model
    #     #TODO CHANGE DENSE LAYER TO 3
    #     pred = Dense(3, activation='softmax')(x)
    #     model = Model(inputs=clip_input, outputs=pred)
    #     return model
    # # build model
    # def get_gate_flow_slow_fast_model(self):
    #     """
    #     build gate_flow_slow_fast without weight_steals
    #     :return: gate_flow_slow_fast model
    #     """
    #     model = self.gate_flow_slow_fast_network_builder()
    #     return model
    # #build_abuse_AND_fall_models+weight_steals
    # def build_shoplifting_net_models(self):
    #     self.shoplifting_model = self.get_gate_flow_slow_fast_model()
    #     self.shoplifting_model.load_weights(self.weight_path_Shoplifting)
    #     print("[+][+]download Shoplifting model and weight_steals")
    #
    # #
    # def get_new_model_shoplifting_net(self):
    #     self.shoplifting_model = self.ShopliftingNet_RGB.load_model_and_weight()
    #
    #
    # def get_abuse_model_and_weight_json(self):
    #     #read model json
    #     # load json and create model
    #     json_path = r"E:\FINAL_PROJECT_DATA\[1]ADS_DEMO\model_Abuse_at_epoch_3_28_7_21_round2.json"
    #     json_file = open(json_path, 'r')
    #     loaded_model_json = json_file.read()
    #     json_file.close()
    #     self.shoplifting_model = model_from_json(loaded_model_json)
    #     # load weights into new model
    #     self.shoplifting_model.load_weights(self.weight_path_abuse)
    #
    #     #print("Loaded EMS model,weight_steals from disk")

    ###



    # def getOpticalFlow(self,frames):
    #     """Calculate dense optical flow of input video
    #     Args:
    #         frames: the input video with shape of [frames,height,width,channel]. dtype=np.array
    #     Returns:
    #         flows_x: the optical flow at x-axis, with the shape of [frames,height,width,channel]
    #         flows_y: the optical flow at y-axis, with the shape of [frames,height,width,channel]
    #     """
    #     # initialize the list of optical flows
    #     gray_video = []
    #     for i in range(len(frames)):
    #         img = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
    #         gray_video.append(np.reshape(img, (224, 224, 1)))
    #
    #     flows = []
    #     for i in range(0, len(frames) - 1):
    #         # calculate optical flow between each pair of frames
    #         flow = cv2.calcOpticalFlowFarneback(gray_video[i], gray_video[i + 1], None, 0.5, 3, 15, 3, 5, 1.2,
    #                                             cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    #         # subtract the mean in order to eliminate the movement of camera
    #         flow[..., 0] -= np.mean(flow[..., 0])
    #         flow[..., 1] -= np.mean(flow[..., 1])
    #         # normalize each component in optical flow
    #         flow[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
    #         flow[..., 1] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
    #         # Add into list
    #         flows.append(flow)
    #
    #     # Padding the last frame as empty array
    #     flows.append(np.zeros((224, 224, 2)))
    #     # print("in optical flow {}".format(np.array(flows, dtype=np.float32).shape))
    #     # return np.array(flows, dtype=np.float32)
    #     return np.array(flows)

    def uniform_sampling(self,np_video_frame, target_frames=64):
        # get total frames of input video and calculate sampling interval
        len_frames = int(len(np_video_frame))
        interval = int(np.ceil(len_frames / target_frames))

        # init empty list for sampled video and
        sampled_video = []
        # step over np video frames list with and append to sample video at each interval step
        # sample_video is equal to (64,224,224,5)
        # extract  (64,224,224,5) frame  from np_video_frame at size(149,224,224,5)
        for i in range(0, len_frames, interval):
            # print("i={}\nnp_video_frame[i].shape={}".format(i,np.array(np_video_frame[i]).shape))
            # exit()
            sampled_video.append(np_video_frame[i])
            # calculate numer of padded frames and fix it
        num_pad = target_frames - len(sampled_video)
        padding = []
        if num_pad > 0:
            for i in range(-num_pad, 0):
                try:
                    padding.append(np_video_frame[i])
                except:
                    padding.append(np_video_frame[0])
            sampled_video += padding
            # get sampled video
            ###
            sampled_video = np.array(sampled_video)
            ####
            # print("this is the acutle input---:{}\nexit...".format(sampled_video.shape))
            # exit()
        return np.array(sampled_video, dtype=np.float32)

    def normalize(self,data):
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std

    # step1 make frame format
    # dont use in this case
    def make_frame_set_format(self, frame_set_src, resize=(224, 224)):
        """
        this function gets frame set video and risize it 224,224
        :param frame:
        :return:frame set List format
        """
        frame_set = []
        for frame in frame_set_src:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.reshape(frame, (224, 224, 3))
            frame_set.append(frame)
        return np.array(frame_set)
    def make_frame_format(self,frame,resize=(224,224)):
        """
        :param frame:
        :return:frame format that sout the model input
        """
        frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.reshape(frame, (224, 224, 3))
        return frame
    #
    # # step2 get optical flow of the frame
    # def frame_preprocessing(self,frames):
    #     """
    #     get the optical flow and uniform_sampling and normalize
    #     :param frames: list of frames in size (149,224,224,5)
    #     :return: np array to predction in size(-1,64,224,224,5)
    #     """
    #     #frames = np.array(self.frames)
    #     # get the optical flow
    #     #flows = self.getOpticalFlow(frames)
    #     # len_flow size is 149
    #     #result = np.zeros((len(flows), 224, 224, 5))
    #     result = frames
    #     #result[..., 3:] = flows
    #
    #     # unifrom sampling return np array(49,224,224,5)
    #     result = self.uniform_sampling(np_video_frame=result, target_frames=64)
    #
    #     # normalize rgb images and optical flows, respectively
    #     result[..., :3] = self.normalize(result)
    #     #result[..., 3:] = self.normalize(result[..., 3:])
    #
    #     result = result.reshape((-1, 64, 224, 224, 3))
    #     return result
    #
    # # step3 make predecion on the frame after preproccisng
    # def frame_prediction(self, frame_pred):
    #     predictions = self.shoplifting_model.predict(frame_pred)
    #     predictions = predictions[0]
    #     fight = predictions[0]
    #     not_fight = predictions[1]
    #     #print(f'in frame prediction1 \nfight:{fight}\nnot fight:{not_fight}\n')
    #     #print("FIGHT: " + "{:.2f}\n".format(fight))
    #     #print("NOT FIGHT: " + "{:.2f}\n".format(not_fight))
    #     fight = fight.item()
    #     not_fight = not_fight.item()
    #     #print(f'in frame prediction2 \nfight:{round(fight, 4)}\nnot fight:{round(not_fight, 4)}\n')
    #     print("FIGHT: " + "{:.2f}\n".format(fight))
    #     #print("NOT FIGHT: " + "{:.2f}\n".format(not_fight))
    #     return [round(fight, 3), round(not_fight, 3)]
    #
    # def run_frames_check(self, frames, test_index):
    #     """
    #     use this function when we want to make prediction on frames set
    #     :param frames:frames to check when she size is (149,224,224,5)
    #     :param test_index:number of test
    #     :return: [fight , not_fight , bool] fight,not_fight are the prediction probability
    #     """
    #
    #     self.test_index = test_index
    #     self.frames = frames
    #     print("##CHECK NUMBER {}\n\tSTART TIME:{}\n".format(test_index, self.get_time_stemp()))
    #     # get frame after calc optical flow
    #     RES_TO_PREDICT = self.frame_preprocessing()
    #     # get model prediction
    #     fight, not_fight = self.frame_prediction(RES_TO_PREDICT)
    #     print("\t##DONE FRAMES CHECK {}\nFIGHT:%{}\nNOT FIGHT:%{}\n".
    #           format(self.get_time_stemp(), (fight) * 100, 100 * not_fight))
    #
    #
    #
    #           #format(self.get_time_stemp(), (fight) , not_fight))
    #
    #
    #     if(fight >  not_fight):
    #         return[fight,not_fight,True,self.test_index]
    #
    #     elif(fight < not_fight):
    #         return [fight,not_fight,False,self.test_index]
    #
    #     else:
    #         print("FIGHT == NOT FIGHT\nCONSIDER THIS AS FIGHT FOE NOW\n")
    #         return [fight, not_fight, True,self.test_index]

    def ShopLifting_frame_prediction(self, frame_pred):
        """
        This functions get np frame set with optical flow calculate
        and get prediction from ADS model
        :param frame_pred:
        :return: list  = [round(fight, 3), round(not_fight, 3)]
        """
        predictions = self.shoplifting_model.predict(frame_pred)
        predictions = predictions[0]

        Bag = predictions[0]
        Clotes = predictions[1]
        Normal = predictions[2]

        Bag = Bag.item()
        Clotes = Clotes.item()
        Normal = Normal.item()
        #print(round(Bag, 3), round(Clotes, 3), round(Normal, 3))
        return [round(Bag, 3), round(Clotes, 3), round(Normal, 3)]

    def help_func_pred(self,pred):
        # return state report [event,not_event,status]
        Bag = pred[0]
        Clotes = pred[1]
        Normal =pred[2]


        if (Normal < Bag and Normal < Clotes):
            if (Bag>Clotes):
                index = 0
            else:
                index  =1

            return [Bag, Clotes, Normal, True, index]
        else:
            index  = 2
            return [Bag, Clotes, Normal, False,index]

    def save_frame_set_after_pred_live_demo(self, EMS_event_path, EMS_event_frame_set, index, pred, flag, w, h):

        #FOR SIMCAM 1
        #file_name = "EMS_event_record_" + str(index) + "__.mp4"
        #fourcc = cv2.VideoWriter_fourcc(*'X264')

        #2
        file_name = "Shoplifting_event_record_" + str(index) + "__.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        #3
        # fourcc = cv2.VideoWriter_fourcc(*'X264')
        # file_name = "EMS_event_record_" + str(index) + "__.mp4"

        video_dst_path = os.path.join(EMS_event_path, file_name)
        # print(f"Final path = {video_dst_path}\nindex = {index}\n")


        out = cv2.VideoWriter(video_dst_path, fourcc, 15, (w, h))

        for frame in EMS_event_frame_set:
            cv2.putText(frame, "Theft alert ", (int(20), int(80)), 0, 5e-3 * 200, (0, 255, 0),3)
            cv2.putText(frame, "Bag: %" + str(round(pred[0] * 100, 4)), (int(20), int(120)), 0, 5e-3 * 200,
                        (0, 255, 0), 3)
            #220, 20, 60 #(0, 255, 0), 3)
            #Hides an item
            cv2.putText(frame, "Clothes: %" + str(round(pred[1] * 100, 4)), (int(20), int(160)), 0, 5e-3 * 200,
                        (0, 255, 0), 3)

            out.write(frame)
        out.release()
        return video_dst_path

    def run_ShopLifting_frames_check(self, frame_set_format_r, index):
        """
        use this function when we want to make prediction on frames set
        :param frames1:bluuer frames to check when she size is (149,224,224,5)
        :param test_index:number of test
        :return: [fall , not_fall , bool]\[fight , not_fight , bool]
        fall\fight,not_fall\fight are the prediction probability
        """
        # todo change back to return fall model after training
        # print("##CHECK NUMBER {}\n".format(index))
        # get frame after calc optical flow
        RES_TO_PREDICT = self.frame_preprocessing(frame_set_format_r)
        # get model prediction
        # ABUSE
        shopLifting_pred = self.ShopLifting_frame_prediction(RES_TO_PREDICT)
        # FALL
        # fall_pred = self.fall_frame_prediction(RES_TO_PREDICT)

        shopLifting_res = self.help_func_pred(shopLifting_pred)
        # fall_res = self.help_func_pred(fall_pred)
        # TODO RETURN THE BIGEST PRED BETWEEN ABUSE AND FALL
        return shopLifting_res

    def run_Shoplifting_frames_check_live_demo(self, frame_set_format_r, ABUSE_flag):
        """
        use this function when we want to make prediction on frames set
        :param frames1:bluuer frames to check when she size is (149,224,224,5)
        :param frames2:rgb frames to extract OpticalFlow (149,224,224,5)
        :param test_index:number of test
        :param flag event condition
               flag==0 check fall,
               flag==1 check abuse,
               flag ==2 check fall and abuse
        :return: [fight , not_fight , bool] fight,not_fight are the prediction probability
        """
        ############


        # print(len(frame_set_format_r))
        # print(type(frame_set_format_r))
        # print(len(frame_set_r))

        # reports = self.run_EMS_frames_check(frame_set_format_r, index)
        # 1
        s_1 = frame_set_format_r[0:64]
        s_2 = []
        s_3 = []

        reports_1 = self.run_EMS_frames_check(s_1, ABUSE_flag)
        # 2
        if (len(frame_set_format_r) - 64 > 0):
            s_2 = frame_set_format_r[64:128]

            reports_2 = self.run_EMS_frames_check(s_2, ABUSE_flag)
        else:
            reports_2 = reports_1
        # 3
        if (len(frame_set_format_r) - 128 > 64):
            s_3 = frame_set_format_r[128:]
            reports_3 = self.run_EMS_frames_check(s_3, ABUSE_flag)

        elif (len(frame_set_format_r) - 128 < 64):
            # print("in_cond EMS_run_model_pipeline")
            last = len(frame_set_format_r) - 128
            res = 64 - last
            # print(f"res= {res}")
            s_3 = frame_set_format_r[128 - res:]
            reports_3 = self.run_EMS_frames_check(s_3, ABUSE_flag)
        else:
            reports_3 = reports_2

        # print(f"s_1-{len(s_1)}\ns_2-{len(s_2)}\ns_3-{len(s_3)}\n")
        #print(f"reports:1-{reports_1}\nreports:2-{reports_2}\nreports:3-{reports_3} ")

        # case found abuse event in one of the samples
        if (reports_1[2] == True or reports_2[2] == True or reports_3[2] == True):
            # todo take the max pred from all samepls
            arr_max = [reports_1[0], reports_2[0], reports_3[0]]
            arr_min = [reports_1[1], reports_2[1], reports_3[1]]
            max = np.max(arr_max)
            min = np.min(arr_min)
            reports = [max,min,True]
            print(colored(f"reports {max,min}",'green'))
            #return [arr_max,arr_min,True]
            # print(f"Max_res={max}")
            # if reports_1[2]:
            #     reports = reports_1
            # elif reports_2[2]:
            #     reports = reports_2
            # else:
            #     reports = reports_3

        # case where we dont found abuse event,
        # return reports_1 or somtihng else maybe bollean
        else:
            reports = reports_1

        return reports

    def split_frame_set(self,frame_set_format):
        """
        return list of frame set 64 frame each
        :param frame_set_format_r:
        :return: list[s1,s2,s3..]
        """
        iter = np.ceil(len(frame_set_format)/64)
        #print(iter)
        set_list = []
        index = 0
        start = 0
        end = 64

        while iter>=0:
            s = frame_set_format[start:end]
            #print(f"iter= {iter}\nlen(s)= {len(s)}")
            set_list.append(s.copy())
            start = start + 32
            end = end +32
            iter = iter - 1

        #print(f"set list len = {len(set_list)}")
        #print("start = {} end = {}".format(start,end))
        return set_list

    def split_frame_set_Recursive(self, frame_set_format):
        """
        return list of frame set 64 frame each
        :param frame_set_format_r:
        :return: list[s1,s2,s3..]
        """
        iter = np.ceil(len(frame_set_format) / 64)
        print(iter)
        set_list = []
        index = 0
        start = 0
        end = 64

        while iter >= 0:
            s = frame_set_format[start:end]
            print(f"iter= {iter}\nlen(s)= {len(s)}")
            set_list.append(s.copy())
            start = start + 32
            end = end + 32
            iter = iter - 1
            # 0:64
            # 32:96
            #
        print(f"set list len = {len(set_list)}")
        print("start = {} end = {}".format(start, end))
        return set_list

    def check_score(self,report):

        print("in check_score, report = {}".format(report))
        if report[4] == 0:
            self.Bag_count = self.Bag_count + 1
        elif report[4] == 1:
            self.Clotes_count = self.Clotes_count + 1
        elif report[4] == 2:
            self.Normal_count = self.Normal_count + 1
        else:
            print(f"[-][-] ERROR in check_score")

    def run_Shoplifting_frames_check_live_demo_2_version(self, frame_set_format_r, Shoplifting_flag):
        """
        use this function when we want to make prediction on frames set
        :param frames1:bluuer frames to check when she size is (149,224,224,5)
        :param frames2:rgb frames to extract OpticalFlow (149,224,224,5)
        :param test_index:number of test
        :param flag event condition
               flag==0 check fall,
               flag==1 check abuse,
               flag ==2 check fall and abuse
        :return: [fight , not_fight , bool] fight,not_fight are the prediction probability
        """
        ############



        res = self.split_frame_set(frame_set_format_r)
        reports = [0,0,0,False,None]
        self.Bag_count = 0
        self.Clotes_count = 0
        self.Normal_count = 0
        for f_set in res:
            reports = self.run_ShopLifting_frames_check(f_set, Shoplifting_flag)
            self.check_score(reports)
            #print("Bag_count={}\nClotes_count={}\nNormal_count={}".format(self.Bag_count,self.Clotes_count,self.Normal_count ))
            # if (reports[3]):
            #     return reports

        if self.Normal_count <= self.Clotes_count and self.Normal_count <= self.Bag_count:
            print("Bag_count={}\nClotes_count={}\nNormal_count={}".format(self.Bag_count, self.Clotes_count,
                                                                            self.Normal_count))
           # return reports
        return reports






