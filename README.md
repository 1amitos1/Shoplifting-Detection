# Shoplifting-Detection
---
#### Project creators
- Amit Hayun [amitos684@gmail.com](amitos684@gmail.com)
- Amit Hayun [Linkedin](https://www.linkedin.com/in/amithayun82797614a/)



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
- A new novel approach in human action recognition (HAR) that combine the advantage of deep neural network **3D-CNN** in learning complex features and patterns, and lightweight models architecture suitable for real-time interference and edge devices deployment.
- SL models are built using a unique dataset designed explicitly for convenience stores and supermarkets containing row videos of theft events performed by actors
- model designed for use in real-time environments using surveillance
  cameras and deliver real-time alerts on theft incidents occurring in the store.





## Introduction 
---

### Computer vision and Human action recognition quick overview
- Artificial intelligence and recent innovations in deep learning (DL) and neural networks took great leaps in recent years. One of the most effective types of AI is computer vision. Computer vision is the field of computer science that focuses on replicating parts of the complexity of the human vision and enabling computers to identify and process objects in images in a similar way has humans accomplish.
One of the most challenging tasks in computer vision is Human Action Recognition (HAR) which aims to understand human behavior and assign a label to each action.

- A common approach in the HAR domain, is to treat video frames as images and apply **2D-CNN** (Conventional Neural Network) to recognize the action in every single frame from the video stream.One of the main deficiencies of this approach is that each video frame forms only a tiny part of the video's story, such an approach will be using incomplete information and could, therefore, easily wrongly classified, especially if there are fine-grind distinctions. In the HAR domain a single frame analysis may interoperate as different activity from the realty, when there are different shapes (e.g., person is doing pushups, or he fall down). This is termed interclass similarity, which is a common phenomenon in HAR.

- the solution for interclass similarity problem is to extend the2D- CNN form 2D to 3D. As such the network has access not only the visual appearance present in each video frame, but also the temporal evaluation across consecutive frames. the training of **3D CNN** is very computationally expensive, and the model size also has quadratic growth compared to 2D CNN. The general trend has been to make deeper and more complicated network to achieve higher accuracy. however, these advances to improve accuracy are not necessarily making networks more efficient with respect to size and speed. Despite the effectiveness of **3D-CNN** volumetric models, most of the current architectures requires a huge computational power due to the convolution operation and the large number of parameters. Because of the volumetric nature of the convolutional filters to extract the features which increase the computational complexity cubically, which limits the usage of 3D-CNN volumetric in DL models for real time inference tasks.

- Applications that relay on recognition task to be carried in a timely fashion need to understand the computation limitation of the used platform.
This has special importance when we desire to identify an incident that consists of several complex action, in which the time is critical element, and the observed behavior are complex actions that require prolonged and consistent observation to decide whether an incident occurred or not. Accurate and distinctive algorithms need to be designed to solve this problem.




### Problem statement
Real-time analysis of each camera has become an exhaustive task due to human limitations. The primary human limitation is the visual focus of attention. The human gaze can only concentrate on one specific point at once. Although there are large screens and high-resolution cameras, a person can only regard a small segment of the image at a time. Thieves are well aware that watching all the video footage is too demanding for "SMBs" such as retailers\ grocery\convenience stores, which makes the technology lose its role as a deterrent.

 - **Shoplifting is conservatively estimated to account for 30% – 40% of total retail shrink/losses.**
   + **(Universities of Florida and Hayes International surveys)**
 - **The average shoplifting case value in 2021 was $310.11, reflecting an increase of 13.0% from 2020**
 
#### Project goals:
provides a comprehensive solution for monitoring and detecting unusual events in real-time without the need for human supervision, the system will alert on
a variety of scenarios. **The following example describes the chain of events in the case of a
shoplifting incident, where the customer steals an alcoholic beverage and hides it in a bag**.

![sl_proecess_1](https://user-images.githubusercontent.com/34807427/171988455-913c721b-92ae-4f61-91fe-32eee77b5989.png)


When one of these actions will detected by our AI model, we will provide the store owner with an
immediate alert.

The system monitors basic customer activities in the store, activities that we will define as pre-
crime such as:
- Taking an item off the shelf.
- Returning an item to the shelf.
- Examining an item.
These activities are routine customer procedures in the store.


And activities that we will define as crime lapse in which our system will monitor and report in real-
time on a case of theft in the store.
crime lapse activities such as:
- Concealing an item in clothes.
- Concealing an item in a bag.

**When one of these actions will be detected by our AI model, we will provide the store owner with an
immediate alert**.


## Data 
---
#### Data collection
In order to train deep learning models, the first step is data collection raw video data collected from security cameras from two supermarkets, the theft was
committed by actors in several different theft scenarios inside the store

Scenarios tested:
- Product taking/returning/ examination of the product
- Inserting an item in a pocket / coat / bag


All the cases of theft were examined in a variety of shooting angles, and by rotation of actors and clothing.

we collect 4000 video clips after the filtering process.
A link to the dataset sample is provided for the entire Dataset, send email to amitos684@gmail.com 


#### filtering process
We work according to a machine learning methodology
1. search abuse video links online
2. download the links
3. convert the video to AVI format with FBS=30sec
4. cut the video into 5-sec clips
5. manual extracting from each video 5sec clips [3,4 clips for each video]
6. create more videos by using 5 data argumentation techniques

 <img src="https://user-images.githubusercontent.com/34807427/117050368-f15d1c00-ad1d-11eb-85eb-d21343f74e55.png" width="300" height="300">


## Model architecture
---
##### Network name: **Gate_Flow_SlowFast**

#### Model description:
 Inspired by **SlowFast Networks for Video Recognition** and the **mobileNet-SSD** architecture.
 this Model design combines the Tow gate stream architecture and the SlowFastNetwork architecture.
The idea is to simulate the human brain in the aspect of visual information processing and split the data into 2 channels.

- **Slow**
 Simulates a slow information processing process in the brain - the goal of this channel is to teach the network as deep characteristics as possible.
Receives as input - 4 Frames

- **Fast**
Simulates a fast information processing process in the brain - the purpose of this channel is to teach the network local properties temporal feature
Receives as input - 64 Frames and within this channel, there are 2 additional sub-channels of **RGB, Optical Flow**
Receives as input - 64 Frames
In the Fast-RGB channel, a Lateral connection used to connect properties to the Slow channel 
![Slow_fast](https://user-images.githubusercontent.com/34807427/172115316-cd57d6c7-4d2e-45a0-8f8e-a6373c6ddb31.png)




For an in-depth understanding of the topic, I suggest reading the original paper 

[SlowFast Networks for Video Recognition](https://scontent.ftlv7-1.fna.fbcdn.net/v/t39.8562-6/240838925_377595027406519_956785818926520821_n.pdf?_nc_cat=103&ccb=1-7&_nc_sid=ad8a9d&_nc_ohc=7as3khAgb1QAX9fsxcb&_nc_ht=scontent.ftlv7-1.fna&oh=00_AT9RK1GZmt8SrepxHyqL1c8iyQxtaNOW3GXccaw51aQyww&oe=62A0E274)


<img src="https://user-images.githubusercontent.com/34807427/172118022-8e5578ab-fb67-4266-973b-6353fb9b895c.png" width="450" height="300">


| Total params        |Trainable params   | Non-trainable  |
| ------------- |:-------------:| -----:|
| 500,386      | 500,386 | 0 |

The model architecture is based on mobileNet SSD.
And the highlight of this model is utilizing  tow path
Slow and Fast, and for each path, there are tow channel one for optical flow and one for RGB channel.


<img src="https://user-images.githubusercontent.com/34807427/172116117-a53c1512-dde7-4d6f-9fc4-57177a0dc0e7.jpg" width="850" height="700">


- Conv3D split into two channels -  RGB frame and Optical flows as shown in the figure below.
- Relu activation is adopted at the end of the RGB channel. 
- Sigmoid activation is adopted at the end of the Optical flow channel.
- RGB and Optical Flow channels outputs are multiplied together and processed by a temporal max-pooling.
- Merging Block is composed of basic 3D CNNs, used to process information after self learned temporal pooling. 
- Fully-connected layers generate output.

 <img src="https://user-images.githubusercontent.com/34807427/117047169-3c753000-ad1a-11eb-93a5-7825120596ca.png" width="550" height="400">
 
 <img src="https://user-images.githubusercontent.com/34807427/171699014-2f4c0d51-662f-42fc-b2b9-4c8b9e2b1d43.png" width="750" height="600">



## Model training && Evaluation

![צילום מסך 2022-06-04 121433](https://user-images.githubusercontent.com/34807427/171992905-bed95bdc-204f-40ef-9df2-825e8288b82e.png)
Examination of the model on our dataset we received an accuracy of 85.77%
Compared to the SlowFast model we got the following results 76%
![ee](https://user-images.githubusercontent.com/34807427/171993248-347f44dd-44fb-4402-8b02-30a527afd2c1.png)




## Input-Output
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

