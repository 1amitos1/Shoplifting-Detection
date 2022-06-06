import cv2
import queue
import time
import threading
import numpy as np
from termcolor import colored

from Shoplifting import Alert
from Shoplifting.data_pip_shoplifting import Shoplifting_Live
import warnings
# warnings.filterwarnings("ignore")
# warnings.simplefilter(action='error', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=FutureWarning)

# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD
from datetime import date,datetime
#from datetime import datetime
import tensorflow as tf
from keras.models import Input, Model
from tensorflow.keras.optimizers import SGD, Adam
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, Multiply,Add,Concatenate
from keras.layers.core import Lambda
from keras.models import model_from_json


# from object_detection.utils import label_map_util
# from object_detection.utils import config_util
# from object_detection.utils import visualization_utils as viz_utils
# from object_detection.builders import model_builder

def get_abuse_model_and_weight_json():
    # read model json
    # load json and create model
    weight_abuse = r"E:\FINAL_PROJECT_DATA\2021\Silence_Vision__EDS_Demo\Event_detection\Event_weight\Abuse\weights_at_epoch_3_28_7_21_round2.h5"
    json_path = r"E:\FINAL_PROJECT_DATA\2021\Yolov5_DeepSort_Pytorch-master\Yolov5_DeepSort_Pytorch-master\EMS\model_Abuse_at_epoch_3_28_7_21_round2.json"
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    abuse_model = model_from_json(loaded_model_json)
    # load weights into new model
    abuse_model.load_weights(weight_abuse)
    print("Loaded EMS model,weight_steals from disk")
    return abuse_model

# ABUSE_MODEL = get_abuse_model_and_weight_json()
q = queue.Queue(maxsize=3000)
frame_set = []


Frame_set_to_check = []
Frame_INDEX = 0
lock = threading.Lock()
Email_alert_flag = False
email_alert = Alert.Email_Alert()
shoplifting_SYS = Shoplifting_Live()
W=0
H=0
src_main_dir_path =r"E:\FINAL_PROJECT_DATA\2021\ALL_DATA_PART_2\ALL_IMPORTENT DESKTOP\DESKTOP_22_9_21\PROJECT_FILE\test_shoplifting"
#src_main_dir_path = r"C:\Users\amit hayoun\Desktop\test3\3\aaa.avi"



def Receive():
    global H,W
    #print("start Receive")
    #rtsp://SIMCAM:2K93AG@192.168.1.2/live
    #video_cap_ip = 'rtsp://SIMCAM:S6BG9J@192.168.1.20/live'
    #video_cap_ip = r'rtsp://barloupo@gmail.com:ziggy2525!@192.168.1.9:554/stream2'
    video_cap_ip= r"E:\FINAL_PROJECT_DATA\2021\ALL_DATA_PART_2\ALL_IMPORTENT DESKTOP\DESKTOP_22_9_21\PROJECT_FILE\test_shoplifting\org_video\The Absolut Abolisher1977_.avi"
    cap = cv2.VideoCapture(video_cap_ip)
    # cap.set(3, 640)
    # cap.set(4, 480)
    W = int(cap.get(3))
    H = int(cap.get(4))
    #print("H={}\nW={}".format(H,W))
    ret, frame = cap.read()
    print(colored(ret, 'green'))
    q.put(frame)
    #while cap.isOpened():
    while ret:
        ret, frame = cap.read()
        q.put(frame)


def Display():
    global Frame_set_to_check,Frame_INDEX
    print(colored('Start Displaying', 'blue'))

    while True:
        if q.empty() != True :
            frame = q.get()
            if isinstance(frame, type(None)):
                print("[-][-] NoneType frame {}".format(type(frame)))
                break

            frame_set.append(frame.copy())
            #print(len(frame_set))
            if len(frame_set) == 149:
                Frame_set_to_check = frame_set.copy()

                #print(type(Frame_set_to_check))
                #p3 = threading.Thread(target=Pred)
                Pred()
                time.sleep(1)
                frame_set.clear()

            #cv2.imshow("frame1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def Pred():
    global Frame_set_to_check, Frame_INDEX
    #ems = EMS_Live()
    with lock:
        #RGB + OPT NET
        #shoplifting_SYS.build_shoplifting_net_models()

        #RGB NET ONLY
        shoplifting_SYS.load_model_and_weight_gate_flow_slow_fast_RGB()

        Frame_set_to_check_np = np.array(Frame_set_to_check.copy())

        Frame_set = shoplifting_SYS.make_frame_set_format(Frame_set_to_check_np)

        reports = shoplifting_SYS.run_StealsNet_frames_check_live_demo_2_version(Frame_set, Frame_INDEX)
        #print(reports)
        Frame_INDEX = Frame_INDEX + 1
        ##
        Bag = reports[0]
        Clotes = reports[1]
        Normal = reports[2]
        state = reports[3]
        #todo event_index maybe paas a dict
        event_index = reports[4]
        #print("event_index {}".format(event_index))
        ##


        if (state):
            print(colored(f"---------------------", 'green'))
            print(colored('Found shopLifting event', 'green'))
            print(colored(f"Bag: {Bag}\nClotes: {Clotes}\nNormal: {Normal}", 'green'))
            #print(colored(f"reports {reports[0], reports[1],reports[2]}", 'green'))
            print(colored(f"Test number:{Frame_INDEX-1}\n---------------------\n", 'green'))
            # print("fight:{}\nnot fight:{}".format(fight,not_fight))

            prob = [Bag, Clotes,Normal]

            found_fall_video_path = shoplifting_SYS.save_frame_set_after_pred_live_demo(src_main_dir_path,
                                                                                 Frame_set_to_check,
                                                                                 Frame_INDEX-1, prob,
                                                                                 0, W, H)


            if Email_alert_flag:
                file_name = found_fall_video_path.split("\\")[-1]
                print(f"path = to email{found_fall_video_path}")
                print(f"file name: {file_name}")
                absulutefilepath = found_fall_video_path
                email_alert.send_email_alert(email_alert.user_email_address3, file_name,
                                                  absulutefilepath)

        else:
            print(colored(f"---------------------", 'red'))
            print(colored("Normal event", 'red'))
            print(colored(f"Test number:{Frame_INDEX - 1}\n---------------------\n", 'red'))
            Frame_set_to_check.clear()

        #lock.release()
        time.sleep(1)



if __name__ == '__main__':
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    #p3 = threading.Thread(target=Pred)
    p1.start()
    p2.start()
    #p3.start()