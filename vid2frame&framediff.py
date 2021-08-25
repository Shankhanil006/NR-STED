import tensorflow as tf
from keras.applications import VGG19
from keras.applications.resnet50 import ResNet50
from keras import models
from keras.layers  import Dense,Flatten,Dropout,GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda, Input, BatchNormalization, Conv2D, MaxPool2D
from keras import optimizers
from keras import backend as K
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.layers import Layer,concatenate, PReLU, LeakyReLU
from tensorflow.keras.layers import Cropping2D
from keras.preprocessing.image import load_img, img_to_array
import glob
import cv2
from scipy import io as sio
import numpy as np
import os
#from cropping import *
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import glob
from operator import itemgetter
from sklearn import datasets, svm
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import SVR, LinearSVR
from sklearn import linear_model
from scipy.stats import spearmanr as srocc
import keras
from keras.optimizers import Adam
from keras.models import load_model
#layer_index = -1
#img_shape = (500,500,3)
from keras import regularizers
import h5py
from keras import losses

from scipy.io import loadmat
import skvideo.io 

from sklearn import linear_model
from sklearn.kernel_approximation import Nystroem
from itertools import combinations
from  scipy.stats import pearsonr as lcc
from scipy.stats import norm
import matplotlib.pyplot as plt
import gc
gc.collect()
import time
from PIL import Image
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
K.clear_session()

    
################# LIVE VQA ###########################################################
def liveVQA(directory):
#    directory = '/media/ece/DATA/Shankhanil/VQA/live_vqa/'
    with open(directory + 'live_video_quality_seqs.txt') as f:
        video_names = f.readlines()
    live_video_list = [x.split('.')[0] for x in video_names] 
    
    with open(directory + 'live_video_quality_data.txt') as f:
        video_dmos = f.readlines()
    live_dmos_list = [float(x.split('\t')[0]) for x in video_dmos] 
    live_dmos = np.array(live_dmos_list)
    
    seq = np.array(['pa', 'rb', 'rh', 'tr', 'st', 'sf', 'bs', 'sh', 'mc', 'pr'])
    rate = [25,25,25,25,25,25,25,50,50,50]
    nframes = [250, 250, 250, 250, 250, 250, 217, 500, 500, 500]
    
    seq_id = {0:'pa', 1:'rb', 2:'rh', 3:'tr', 4:'st', 5:'sf', 6:'bs', 7:'sh', 8:'mc', 9:'pr'}
    
    for i in range(len(live_video_list)):
        vid = live_video_list[i] 
      
    #    generate_frames = ppm_video(live_video_list[i] + '.yuv')
        get_video = skvideo.io.vread(directory + 'liveVideo/' +
                                           vid+ '.yuv', 432, 768, as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'})
        
        
        scene = vid[:2]
        dist_type = vid.split('_')[0][2:]
        out_path =  directory + 'live_vqa_frames_gray/distorted/' + scene + '/' 
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        for j in range(0,len(get_video)-1,2):
#            frame      = get_video[j]
            frame_diff = get_video[j].astype(np.int16) - get_video[j+1].astype(np.int16)
            filepath = out_path + scene +  dist_type        
            np.save(filepath + '_' + str(j+1), frame_diff)
        gc.collect()
        print(i)
#    break
        return()
################# LIVE MOBILE ###########################################################
def live_mobile(mobile_directory):
#    mobile_directory = '/media/ece/DATA/Shankhanil/VQA/live_mobile/'
    
    mobile_video_list = sio.loadmat(mobile_directory + 'strred_mobile.mat')['names'].squeeze()
    mobile_seq_id = {0:'bf', 1:'dv', 2:'fc', 3:'hc', 4:'la', 5:'po', 6:'rb', 7:'sd', 8:'ss', 9:'tk'} 
    
    for i in range(len(mobile_video_list)):
        
        vid = str(mobile_video_list[i][0]) 
        get_video = skvideo.io.vread(mobile_directory + 'LIVE_VQA_mobile/' +
                                           vid+ '.yuv', 720, 1280, as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'})    
        scene = vid.split('_')[0]
        out_path =  mobile_directory + 'live_mobile_frames_gray/distorted/' + scene + '/' 
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        for j in range(0,len(get_video)-1,2):
#            frame = get_video[j]
            frame_diff = get_video[j].astype(np.int16) - get_video[j+1].astype(np.int16)
            filepath = out_path + vid        
            np.save(filepath + '_' + str(j+1), frame_diff)
        gc.collect()
        print(i)
#    break
    return()
  
########################## EPFL POLMI #########################################################
def epfl_polimi_4cif(epfl_directory):
#    epfl_directory = '/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/'
    with open(epfl_directory + 'names_scores.txt') as f:
        file_names = f.readlines()
    epfl_video_list = [x.split()[0] for x in file_names] 
    epfl_dmos_list  = [float(x.split()[1]) for x in file_names]
    
    for i in range(len(epfl_video_list)):
        vid = epfl_video_list[i] 
        idx = np.mod(i,12)
        get_video = skvideo.io.vread(epfl_directory + 'online_DB/decoded/4CIF/' +
                                           vid+ '.yuv', 576, 704, as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'})    
        scene = vid.split('_')[0]
        out_path =  epfl_directory + 'epfl_4cif_frames_gray/distorted/' + scene + '/' 
        if not os.path.exists(out_path):
            os.makedirs(out_path)   
        
        for j in range(0,len(get_video)-1,2):
#            frame = get_video[j]
            frame_diff = get_video[j].astype(np.int16) - get_video[j+1].astype(np.int16)
            filepath = out_path + scene +  '_' + str(idx) 
            np.save(filepath + '_' + str(j+1), frame_diff)
        
        gc.collect()
        print(i) 
    return()
    
def epfl_polimi_cif(epfl_directory):
#    epfl_directory = '/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/'
    with open(epfl_directory + 'names_scores.txt') as f:
        file_names = f.readlines()
    epfl_video_list = [x.split()[0] for x in file_names] 
    epfl_dmos_list  = [float(x.split()[1]) for x in file_names]
    
    for i in range(len(epfl_video_list)):
        vid = epfl_video_list[i] 
        idx = np.mod(i,12)
        get_video = skvideo.io.vread(epfl_directory + 'online_DB/decoded/CIF/' +
                                           vid+ '.yuv', 576, 704, as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'})    
        scene = vid.split('_')[0]
        out_path =  epfl_directory + 'epfl_cif_frames_gray/distorted/' + scene + '/' 
        if not os.path.exists(out_path):
            os.makedirs(out_path)   
        
        for j in range(0,len(get_video)-1,2):
#            frame = get_video[j]
            frame_diff = get_video[j].astype(np.int16) - get_video[j+1].astype(np.int16)
            filepath = out_path + scene +  '_' + str(idx) 
            np.save(filepath + '_' + str(j+1), frame_diff)
        
        gc.collect()
        print(i) 
    return()    
#    break
    
################# CSIQ ###########################################################
def csiq(csiq_directory):
#    csiq_directory = '/media/ece/DATA/Shankhanil/VQA/CSIQ/'
    with open(csiq_directory + 'video_subj_ratings.txt') as f:
        file_names = f.readlines()
    csiq_video_list = [x.split('\t')[0].split('.')[0] for x in file_names] 
    csiq_dmos_list  = [float(x.split('\t')[1]) for x in file_names]
    
    awgn = [13,14,15]
    for i in range(len(csiq_video_list)):
        vid = csiq_video_list[i] 
        get_video = skvideo.io.vread(csiq_directory + 'csiq_videos/' +
                                           vid + '.yuv', 480, 832, as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'})    
        scene = vid.split('_')[0]
        dist_type = vid.split('_')[-1]
        
        if int(dist_type) in awgn:
            continue
        
        out_path =  csiq_directory + 'csiq_frames_gray/distorted/' + scene + '/' 
        if not os.path.exists(out_path):
            os.makedirs(out_path)   
        
        for j in range(0,len(get_video)-1,2):
#            frame = get_video[j]
            frame_diff = get_video[j].astype(np.int16) - get_video[j+1].astype(np.int16)
            filepath = out_path + scene +  '_' + dist_type 
            np.save(filepath + '_' + str(j+1), frame_diff)
        
        gc.collect()
        print(i) 
    return()
#    break


################# ECVQ ###########################################################
def ecvq(ecvq_directory):
#    ecvq_directory = '/media/ece/DATA/Shankhanil/VQA/ECVQ/'
    with open(ecvq_directory + 'subjective_scores_cif.txt') as f:
        file_names = f.readlines()
    ecvq_video_list = [x.split()[0] for x in file_names] 
    ecvq_dmos_list  = [float(x.split()[2]) for x in file_names]
    
    for i in range(len(ecvq_video_list)):
        vid = ecvq_video_list[i] 
        get_video = skvideo.io.vread(ecvq_directory + 'cif_videos/' +
                                           vid + '.yuv', 288, 352, as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'})    
        scene = vid.rsplit('_', 1)[0]
        dist_type = vid.rsplit('_', 1)[-1]
        
        out_path =  ecvq_directory + 'ecvq_frames_gray/distorted/' + scene + '/' 
        if not os.path.exists(out_path):
            os.makedirs(out_path)   
        
        for j in range(0,298,2):
#            frame = get_video[j]
            frame_diff = get_video[j].astype(np.int16) - get_video[j+1].astype(np.int16)
            filepath = out_path + scene +  '_' + dist_type 
            np.save(filepath + '_' + str(j+1), frame_diff)
        
        gc.collect()
        print(i) 
    return()
#    break

################## EVVQ ###########################################################
def evvq(evvq_directory):
#    evvq_directory = '/media/ece/DATA/Shankhanil/VQA/EVVQ/'
    with open(evvq_directory + 'subjective_scores_vga.txt') as f:
        file_names = f.readlines()
    evvq_video_list = [x.split()[0] for x in file_names] 
    evvq_dmos_list  = [float(x.split()[2]) for x in file_names]
    
    for i in range(len(evvq_video_list)):
        vid = evvq_video_list[i] 
        get_video = skvideo.io.vread(evvq_directory + 'vga_videos/' +
                                           vid + '.yuv', 480, 640, as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'})    
        scene = vid.rsplit('_', 1)[0]
        dist_type = vid.rsplit('_', 1)[-1]
        
        out_path =  evvq_directory + 'evvq_frames_gray/distorted/' + scene + '/' 
        if not os.path.exists(out_path):
            os.makedirs(out_path)   
        
        for j in range(0, 298, 2):
#            frame = get_video[j]
            frame_diff = get_video[j].astype(np.int16) - get_video[j+1].astype(np.int16)
            filepath = out_path + scene +  '_' + dist_type 
            np.save(filepath + '_' + str(j+1), frame_diff)
        
        gc.collect()
        print(i) 
    return()
    
#    break
database = ''
path_to_videos = ''
#path_to_framediff = ''

if database == 'liveVQA':
    liveVQA(path_to_videos)
elif database == 'live_mobile':
    live_mobile(path_to_videos)
elif database == 'epfl_polimi_cif':
    epfl_polimi_cif(path_to_videos)
elif database == 'epfl_polimi_4cif':
    epfl_polimi_4cif(path_to_videos)
elif database == 'csiq':
    csiq(path_to_videos)
elif database == 'ecvq':
    ecvq(path_to_videos)
elif database == 'evvq':
    evvq(path_to_videos)



