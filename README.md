# Shoplifting-Detection
---
#### Project creators
- Amit Hayun [amitos684@gmail.com](amitos684@gmail.com)

                    

Provide a Deep Learning-Based real-time solution for nursing homes and hospitals for detecting cases of abuse in the elderly population by analyzing security camera frames and performing real-time forecasting using three machine learning models YOLO, DeepSort, ADS
SL -Shoplifting detection Provides real-time alerts for the SMB market retailers, to monitor and report customer behavior when shoplifting occur.
 

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
-	Creating a new novel dataset ADS-dataset that contain worldwide videos
clips of abuse capture by surveillance cameras in the real-world scenes 

-	Design and implemented ML pipeline for video raw data to generate features, data augmentation techniques, and resampled datasets for model training.

-	Build and train a machine learning model[ADS] based on MobileNet-SSD architecture with 3D-CNN and two-stream method [RGB, OPT].
Training and evaluation of the model using AWS-SageMaker and TensorFlow frameworks. Achieved 87% in F1-score on DCSASS Abuse Dataset and 84%  on ADS Dataset.

-	Combine all models to an overall system and deploying the system in the Streamlit web app that enables the user to get real-time notification alerts
when an abuse event capture by the security camera.

## Introduction 
---
This project is defined as research(60%)\development(40%).

- Research 
Build and train deep learning models(according to a standard ML approach) to automatically identify abuse event capture by security camera

- Development
Build prototype system ADS(Abuse detection system) for deploying models and test them in a real-time environment

### Our Main goal - provide an automated solution for detecting cases of abuse of the elderly.

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






https://user-images.githubusercontent.com/34807427/171149238-3cabeffb-1087-4748-b7ca-1927cd4cf6f8.mp4



https://user-images.githubusercontent.com/34807427/171149909-50489465-6fb0-4e61-ad56-927233318259.mp4
