import tensorflow as tf
from keras.applications import VGG19
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras import models
from keras.layers  import Dense,Flatten,Dropout,GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda, Input
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
#from cropping import *
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from operator import itemgetter
from sklearn import datasets, svm
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import SVR
from sklearn import linear_model
from scipy.stats import spearmanr as srocc
import keras
from keras.optimizers import Adam
from keras.models import load_model
#layer_index = -1
#img_shape = (500,500,3)
from keras import regularizers
import h5py
import matplotlib.pyplot as plt
from scipy.io import loadmat
import gc
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import skvideo.io

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
K.clear_session()


base_model = ResNet50(include_top=False, weights='imagenet', input_shape=None, input_tensor=None, pooling='avg')

#directory = '/media/ece/DATA/Shankhanil/VQA/live_mobile/LIVE_VQA_mobile'                # 720x1280
#directory = '/media/ece/DATA/Shankhanil/VQA/live_vqa/liveVideo'                        # 432x768
#directory = '/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/decoded/4CIF'    # 576x704
#directory = '/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/decoded/CIF'     # 288x352
#directory = '/media/ece/DATA/Shankhanil/VQA/CSIQ/csiq_videos'                          # 480x832
#directory = '/media/ece/DATA/Shankhanil/VQA/ECVQ/cif_videos'                           # 288x352
#directory = '/media/ece/DATA/Shankhanil/VQA/EVVQ/vga_videos'                           # 480x640

databases = {  'live'      :{'height': 432, 'width' : 768}, \
               'mobile'    :{'height':720, 'width' :1280}, \
               'epfl_cif'  :{'height':288, 'width' :352},   \
               'epfl_4cif' :{'height':576, 'width' : 704},   \
               'csiq'      :{'height':480, 'width' : 832}, \
               'ecvq'      :{'height':288, 'width' :  352},   \
               'evvq'      :{'height':480, 'width' : 640} }


video_directory = ''
output_dir = ''
if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
names = [file for file in glob.glob(video_directory +'/*.yuv')]
count = 0

for name in names:
    load_video = skvideo.io.vread(video_directory +name, height, width, as_grey = False, inputdict={'-pix_fmt': 'yuvj420p'})
    #    load_video = skvideo.io.vread(name)
    
    video_feat_frame = []
    for frame in range(0,len(load_video)-1,2):      #298 for ecvq & evvq
        video_cur = load_video[frame]#.astype(np.int16)

        video_cur = preprocess_input(np.expand_dims(video_cur.copy(), axis = 0))
        video_feat_frame.append(base_model.predict(video_cur).squeeze())
    video_feat_frame = np.array(video_feat_frame)
    
    video_feat_name = name.split('.yuv')[0]
#    video_feat_name = name.split('/')[-1].split('.yuv')[0]

    np.save(output_dir + video_feat_name, video_feat_frame)
    print(count)
    count = count+1
    gc.collect()
