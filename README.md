# Shoplifting-Detection
---
#### Project creators
- Amit Hayun [amitos684@gmail.com](amitos684@gmail.com)

SL -Shoplifting detection Provides real-time alerts for the SMB market retailers, to monitor and report customer behavior when shoplifting occur, by analyzing security camera frames and performing real-time forecasting using cutting edge lightweight deep neural  network 3D-CNN architecture

## Table of contents
---
* [Project highlights](#Project-highlights)
* [Introduction ](#Introduction)
* [Data collection](#Data-collection)
* [Model architecture](#Model-architecture)
* [Model training && Evaluation](#Model-training-&&-Evaluation)
* [System overview](#System-overview)
* [ADS PIPELINE Demo](#ADS-PIPELINE-Demo)
* [Input-Output examples](#Input-Output-examples)
* [Setup](#setup)
* [Reference](#Reference)


## Project highlights
---
- A new novel approach in human action recognition (HAR) that combine the advantage of deep neural network 3D-CNN in learning complex features and patterns, and lightweight models architecture suitable for real-time interference and edge devices deployment.
- SL models are built using a unique dataset designed explicitly for convenience stores and supermarkets containing row videos of theft events performed by actors
- model designed for use in real-time environments using surveillance
  cameras and deliver real-time alerts on theft incidents occurring in the store.





## Introduction 
---

### Computer vision and Human action recognition quick overview
- Artificial intelligence and recent innovations in deep learning (DL) and neural networks took great leaps in recent years. One of the most effective types of AI is computer vision. Computer vision is the field of computer science that focuses on replicating parts of the complexity of the human vision and enabling computers to identify and process objects in images in a similar way has humans accomplish.
One of the most challenging tasks in computer vision is Human Action Recognition (HAR) which aims to understand human behavior and assign a label to each action.

- A common approach in the HAR domain, is to treat video frames as images and apply 2D-CNN (Conventional Neural Network) to recognize the action in every single frame from the video stream.One of the main deficiencies of this approach is that each video frame forms only a tiny part of the video's story, such an approach will be using incomplete information and could, therefore, easily wrongly classified, especially if there are fine-grind distinctions. In the HAR domain a single frame analysis may interoperate as different activity from the realty, when there are different shapes (e.g., person is doing pushups, or he fall down). This is termed interclass similarity, which is a common phenomenon in HAR.

- the solution for interclass similarity problem is to extend the2D- CNN form 2D to 3D. As such the network has access not only the visual appearance present in each video frame, but also the temporal evaluation across consecutive frames. the training of 3D CNN is very computationally expensive, and the model size also has quadratic growth compared to 2D CNN. The general trend has been to make deeper and more complicated network to achieve higher accuracy. however, these advances to improve accuracy are not necessarily making networks more efficient with respect to size and speed. Despite the effectiveness of 3D-CNN volumetric models, most of the current architectures requires a huge computational power due to the convolution operation and the large number of parameters. Because of the volumetric nature of the convolutional filters to extract the features which increase the computational complexity cubically, which limits the usage of 3D-CNN volumetric in DL models for real time inference tasks.

- Applications that relay on recognition task to be carried in a timely fashion need to understand the computation limitation of the used platform.
This has special importance when we desire to identify an incident that consists of several complex action, in which the time is critical element, and the observed behavior are complex actions that require prolonged and consistent observation to decide whether an incident occurred or not. Accurate and distinctive algorithms need to be designed to solve this problem.




### Problem statement
Real-time analysis of each camera has become an exhaustive task due to human limitations. The primary human limitation is the visual focus of attention. The human gaze can only concentrate on one specific point at once. Although there are large screens and high-resolution cameras, a person can only regard a small segment of the image at a time. Thieves are well aware that watching all the video footage is too demanding for "SMBs" such as retailers\ grocery\convenience stores, which makes the technology lose its role as a deterrent. Shoplifting is conservatively estimated to account for 30% – 40% of total retail shrink/losses. (Universities of Florida and Hayes International surveys)



- Research 
Build and train deep learning models(according to a standard ML approach) to automatically identify abuse event capture by security camera

- Development
Build prototype system ADS(Abuse detection system) for deploying models and test them in a real-time environment


## Data collection
---
Data collection
In order to train deep learning models, the first step is data collection
We build data collection pipe and gather abuse video from the web
we collect 842 video clips after the filtering process


We work according to a machine learning methodology
1. search abuse video links online
2. download the links
3. convert the video to AVI format with FBS=30sec
4. cut the video into 5-sec clips
5. manual extracting from each video 5sec clips [3,4 clips for each video]
6. create more videos by using 5 data argumentation techniques
7. split the data to Train, Val, Test as shown in table2

- Method and DB expleind -[method and DB expleind.pdf](https://github.com/1amitos1/AbuseDetectionSystem_demo/files/6423235/method.and.db.expleind.pdf)

 <img src="https://user-images.githubusercontent.com/34807427/117050368-f15d1c00-ad1d-11eb-85eb-d21343f74e55.png" width="300" height="300">


## Model architecture
---
The model architecture is based on mobileNet SSD.
And the highlight of this model is utilizing
a branch of the optical flow channel to 
help build a pooling mechanism.

- Conv3D split into two channels -  RGB frame and Optical flows as shown in the figure below.
- Relu activation is adopted at the end of the RGB channel. 
- Sigmoid activation is adopted at the end of the Optical flow channel.
- RGB and Optical Flow channels outputs are multiplied together and processed by a temporal max-pooling.
- Merging Block is composed of basic 3D CNNs, used to process information after self learned temporal pooling. 
- Fully-connected layers generate output.

 <img src="https://user-images.githubusercontent.com/34807427/117047169-3c753000-ad1a-11eb-93a5-7825120596ca.png" width="550" height="400">
 
 <img src="https://user-images.githubusercontent.com/34807427/171699014-2f4c0d51-662f-42fc-b2b9-4c8b9e2b1d43.png" width="550" height="700">







https://user-images.githubusercontent.com/34807427/171149238-3cabeffb-1087-4748-b7ca-1927cd4cf6f8.mp4



https://user-images.githubusercontent.com/34807427/171149909-50489465-6fb0-4e61-ad56-927233318259.mp4


## Reference
---
- Yolo_v3 (https://github.com/qqwweee/keras-yolo3)
- DeepSort (https://github.com/nwojke/deep_sort)
- RWF (https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection/blob/master/README.md)


```
@article{RWF-2000,
  author = {Ming Cheng, Kunjing Cai, and Ming Li},
  title={RWF-2000: An Open Large Scale Video Database for Violence Detection},
  year={2019}
}
```


```
@article{DeepSort,
  author = { B. A. &. P. D. Wojke},
  title={Simple online and realtime tracking with a deep association metric},
  year={2017}
}
```

```
@article{YOLOv3,
  author = {Joseph Redmon },
  title={YOLOv3: An Incremental Improvement},
  year={2018}
}
```

