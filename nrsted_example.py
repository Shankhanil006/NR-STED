import tensorflow as tf
from keras.applications import VGG19
from keras.applications.resnet50 import ResNet50
from keras import models
from keras.layers  import Dense,Flatten,Dropout,GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda, Input, BatchNormalization
from keras import optimizers
from keras import backend as K
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.layers import Layer,concatenate, PReLU, LeakyReLU
from tensorflow.keras.layers import Cropping2D
import glob
import cv2
from scipy import io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
#from cropping import *
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from operator import itemgetter
from sklearn import datasets, svm
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import SVR, LinearSVR
from sklearn import linear_model
from scipy.stats import spearmanr as srocc
import keras
from keras.optimizers import Adam, Nadam
from keras.models import load_model
#layer_index = -1
#img_shape = (500,500,3)
from keras import regularizers
import h5py
from keras import losses

from scipy.io import loadmat, savemat

from sklearn import linear_model
from sklearn import preprocessing
from sklearn.kernel_approximation import Nystroem
from itertools import combinations
import time
import skvideo.io 
from argparse import ArgumentParser

K.clear_session()

# The following code can be used to estimate the NR-STED index of a video. For videos in YUV420 format please specify the video width and height.
#The link to spatial and temporal models are provided in readme.md file
#############################################################################
resnet_model   = ResNet50(include_top=False, weights='imagenet', input_shape=None, input_tensor=None, pooling='avg')
spatial_model  = '../spatial_model.h5'
temporal_model = '../temporal_model.h5'

path_to_video = ''

if path_to_video.split('/')[-1].split('.')[-1] == 'mp4':
     rgb_video = skvideo.io.vread(path_to_video, as_grey = False)
     grey_video = skvideo.io.vread(path_to_video, as_grey = True)
     
elif path_to_video.split('/')[-1].split('.')[-1] == 'yuv':
    height = 1280
    width = 1920
    rgb_video = skvideo.io.vread(path_to_video, height, width,
                     as_grey = False, inputdict={'-pix_fmt': 'yuvj420p'})
    
    grey_video = skvideo.io.vread(path_to_video, height, width,
                     as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'})

s_predict_score = []
t_predict_score = []  

for j in range(0,len(rgb_video)-1,2):
    frame = rgb_video[j].astype(np.int16)
    frame_diff = grey_video[j].astype(np.int16) - grey_video[j+1].astype(np.int16)
    
    pred_spatial  = spatial_model.predict(resnet_model.predict(np.expand_dims(frame,0))).squeeze()
    pred_temporal = temporal_model.predict(np.expand_dims(frame_diff,0)).squeeze()   
    
    s_predict_score.append(pred_spatial)
    t_predict_score.append(pred_temporal)


nr_sted = np.mean(s_predict_score) *np.mean(t_predict_score)
