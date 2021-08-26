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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
K.clear_session()

#########################################################################################################    
spatial_feat_dir  = '/media/ece/DATA/Shankhanil/VQA/resnet50_feat/'
strred_directory = '/media/ece/DATA/Shankhanil/VQA/strred_files/'
allvs1_directory = '/media/ece/DATA/Shankhanil/VQA/Allvs1_models/temporal/'
linear_model = '/media/ece/DATA/Shankhanil/VQA/linear_models/'
#temp_model_directory = '/media/ece/DATA/Shankhanil/VQA/1vsAll_models/temporal/'
#temporal_model = load_model(temp_model_directory + 'evvqvsall/' + 'temporal_evvqvsall_36000.h5')
##################################  LIVE VQA  #####################################
#temporal_model = load_model(allvs1_directory + 'allvslive/' + 'temporal_allvslive_23000.h5') #23k
#
#live_directory = '/media/ece/DATA/Shankhanil/VQA/live_vqa/'
#with open(live_directory + 'live_video_quality_seqs.txt') as f:
#    video_names = f.readlines()
#live_video_list = [x.strip('.yuv\n') for x in video_names] 
#
#with open(live_directory + 'live_video_quality_data.txt') as f:
#    video_dmos = f.readlines()
#live_dmos_list = [float(x.split('\t')[0]) for x in video_dmos] 
#live_dmos = np.array(live_dmos_list)
#
#seq = np.array(['pa', 'rb', 'rh', 'tr', 'st', 'sf', 'bs', 'sh', 'mc', 'pr'])
#rate = [25,25,25,25,25,25,25,50,50,50]
#live_s_rred = sio.loadmat(strred_directory + 'strred_live.mat')['srred']
#
#t_predict_score = []
#for i in range(len(live_video_list)):
#    t_predict_score.append([])
#    
#for i in range(len(live_video_list)):
#
#    name = live_video_list[i]
#    scene = np.array(np.where(seq == name[:2])).squeeze()
#    idx = int(name.split('_')[0][2:])
#    
#    get_video = skvideo.io.vread('/media/ece/DATA/Shankhanil/VQA/live_vqa/liveVideo/' +
#                                       name + '.yuv', 432, 768, as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'})
#    
#    for j in range(0,len(get_video)-1,2):
#        frame_diff = get_video[j].astype(np.int16) - get_video[j+1].astype(np.int16)
#               
#        pred_temporal = temporal_model.predict(np.expand_dims(frame_diff.copy(),0)).squeeze()
#              
#        t_predict_score[i].append(pred_temporal.mean())
#
#    print(i)
#pretrained_scores = np.load('/media/ece/DATA/Shankhanil/VQA/tmp/spatial_allvslive_resnet50.npy')
##pretrained_scores = np.load('/media/ece/DATA/Shankhanil/VQA/tmp/spatial_evvqvslive_resnet50.npy')
#
#x1 =[]
#trred = []
#for i in range(len(t_predict_score)):
#    x1.append(pretrained_scores[i] * np.mean(t_predict_score[i]))
#    trred.append(np.mean(t_predict_score[i]))
#z_dmos,_ = srocc(x1, live_dmos)
#
#srred = pretrained_scores
#trred = np.array(trred)
#strred = np.array(x1)
#
#mdic = {"srred":srred, "trred":trred, "strred":strred, "dmos":live_dmos}
##savemat("/media/ece/DATA/Shankhanil/VQA/tmp/allvslive.mat", mdic)
######################### EPFL-4cif #####################################
#temporal_model = load_model(linear_model + 'epfl_high2low/' + 'temporal_4dbvsepfl_48000.h5') #39k
#epfl_4cif_directory = '/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/decoded/4CIF/'
#with open(epfl_4cif_directory + 'names_scores.txt') as f:
#    file_names = f.readlines()
#epfl_4cif_video_list = [x.split()[0] for x in file_names] 
#epfl_4cif_dmos_list  = [float(x.split()[1]) for x in file_names]
#epfl_4cif_dmos = 5-np.array(epfl_4cif_dmos_list)
#
#epfl_4cif_s_rred = sio.loadmat(strred_directory + 'strred_epfl_4cif.mat')['srred']
#
#epfl_4cif_seq = np.array(['CROWDRUN','DUCKSTAKEOFF','HARBOUR','ICE','PARKJOY','SOCCER'])
#
#t_predict_score = []
#for i in range(2*len(epfl_4cif_video_list)):
#    t_predict_score.append([])
#    
#for i in range(len(epfl_4cif_video_list)):
#    
#    name = epfl_4cif_video_list[i]
#    scene = np.array(np.where(epfl_4cif_seq == name.split('_')[0])).squeeze()
#    idx = np.mod(i,12)
#    
#    get_video = skvideo.io.vread('/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/decoded/4CIF/' +
#                                       name + '.yuv', 576, 704, as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'})
#
#    for j in range(0,len(get_video)-1,2):
#        frame_diff = get_video[j].astype(np.int16) - get_video[j+1].astype(np.int16)
#               
#        pred_temporal = temporal_model.predict(np.expand_dims(frame_diff.copy(),0)).squeeze()      
#       
#        t_predict_score[i].append(pred_temporal.mean())
#    print(i)
############################ EPFL-cif #####################################
#epfl_cif_directory = '/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/decoded/CIF/'
#with open(epfl_cif_directory + 'names_scores.txt') as f:
#    file_names = f.readlines()
#epfl_cif_video_list = [x.split()[0] for x in file_names] 
#epfl_cif_dmos_list  = [float(x.split()[1]) for x in file_names]
#epfl_cif_dmos = 5-np.array(epfl_cif_dmos_list)
#
#epfl_cif_s_rred = sio.loadmat(strred_directory + 'strred_epfl_cif.mat')['srred']
#
#epfl_cif_seq = np.array(['foreman','hall','mobile','mother','news','paris'])
#
#for i in range(len(epfl_cif_video_list)):
#    
#    name = epfl_cif_video_list[i]
#    scene = np.array(np.where(epfl_cif_seq == name.split('_')[0])).squeeze()
#    idx = np.mod(i,12)
#    
#    get_video = skvideo.io.vread('/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/decoded/CIF/' +
#                                       name + '.yuv', 288, 352, as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'})
#
#    for j in range(0,len(get_video)-1,2):
#        frame_diff = get_video[j].astype(np.int16) - get_video[j+1].astype(np.int16)
#               
#        pred_temporal = temporal_model.predict(np.expand_dims(frame_diff.copy(),0)).squeeze()
#               
#        t_predict_score[i + len(epfl_4cif_video_list)].append(pred_temporal.mean())  
#    print(i)
#pretrained_scores = np.load('/media/ece/DATA/Shankhanil/VQA/tmp/spatial_tmpvsepfl_resnet50.npy')
##pretrained_scores = np.load('/media/ece/DATA/Shankhanil/VQA/tmp/spatial_evvqvsepfl_resnet50.npy')
#x1 =[]
#trred = []
#for i in range(len(pretrained_scores)):
#    x1.append(np.mean(t_predict_score[i]) * pretrained_scores[i])
#    trred.append(np.mean(t_predict_score[i]))
#    
#epfl_dmos = np.concatenate([epfl_4cif_dmos, epfl_cif_dmos])
#z_dmos,_ = srocc(trred, epfl_dmos)
#
#srred = pretrained_scores
#trred = np.array(trred)
#strred = np.array(x1)

#mdic = {"srred":srred, "trred":trred, "strred":strred, "dmos":epfl_dmos}
##savemat("/media/ece/DATA/Shankhanil/VQA/tmp/allvsepfl.mat", mdic)
############################### MOBILE  LIVE ######################################################
#temporal_model = load_model(allvs1_directory + 'allvsmobile/' + 'temporal_allvsmobile_22000.h5')   #14k
#mobile_directory = '/media/ece/DATA/Shankhanil/VQA/live_mobile/'
#mobile_dmos = sio.loadmat(strred_directory + 'dmos_live_mobile.mat')['dmos'].squeeze()
#
#mobile_s_rred = sio.loadmat(strred_directory + 'strred_mobile.mat')['srred'].squeeze()
#mobile_video_list = sio.loadmat(strred_directory + 'strred_mobile.mat')['names'].squeeze()
#mobile_seq_id = {0:'bf', 1:'dv', 2:'fc', 3:'hc', 4:'la', 5:'po', 6:'rb', 7:'sd', 8:'ss', 9:'tk'}
#
#t_predict_score = []
#for i in range(len(mobile_video_list)):
#    t_predict_score.append([])
#    
#for i in range(len(mobile_video_list)):
#    
#    name = str(mobile_video_list[i][0]) 
#    get_video = skvideo.io.vread('/media/ece/DATA/Shankhanil/VQA/live_mobile/LIVE_VQA_mobile/' +
#                                       name + '.yuv', 720, 1280, as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'})
#    st = time.time()
#    for j in range(0,len(get_video)-1,2):
#        frame_diff = get_video[j].astype(np.int16) - get_video[j+1].astype(np.int16)
#        pred_temporal = temporal_model.predict(np.expand_dims(frame_diff,0)).squeeze()         
#        t_predict_score[i].append(pred_temporal.mean())
#    
#    et = time.time()
#    print(i)
#    break
#pretrained_scores = np.load('/media/ece/DATA/Shankhanil/VQA/tmp/spatial_allvsmobile_resnet50.npy')
##pretrained_scores = np.load('/media/ece/DATA/Shankhanil/VQA/tmp/spatial_evvqvsmobile_resnet50.npy')
#x1 =[]
#trred = []
#for i in range(len(t_predict_score)):
#    x1.append(pretrained_scores[i] * np.mean(t_predict_score[i]))
#    trred.append(np.mean(t_predict_score[i]))
#z_dmos,_ = srocc(x1, mobile_dmos)        
#
#srred = pretrained_scores
#trred = np.array(trred)
#strred = np.array(x1)
#
#mdic = {"srred":srred, "trred":trred, "strred":strred, "dmos":mobile_dmos}
#savemat("/media/ece/DATA/Shankhanil/VQA/tmp/allvsmobile.mat", mdic)
################################  CSIQ  #############################################
#temporal_model = load_model(allvs1_directory + 'allvscsiq/' + 'temporal_allvscsiq_39000.h5')
#csiq_directory = '/media/ece/DATA/Shankhanil/VQA/CSIQ/csiq_videos/'
#with open(csiq_directory + 'video_subj_ratings.txt') as f:
#    file_names = f.readlines()
#csiq_video_list = [x.split('\t')[0].split('.')[0] for x in file_names] 
#csiq_dmos_list  = [float(x.split('\t')[1]) for x in file_names]
#csiq_dmos = np.array(csiq_dmos_list)
#
#csiq_s_rred = sio.loadmat(strred_directory + 'strred_csiq.mat')['srred']
#
#csiq_seq = np.array(['BQMall', 'BQTerrace','BasketballDrive','Cactus', \
#            'Carving','Chipmunks','Flowervase','Keiba','Kimono', \
#            'ParkScene','PartyScene','Timelapse'])
#
#t_predict_score = []
#for i in range(180):
#    t_predict_score.append([])
#    
#awgn = [13,14,15]
#count = 0
#for i in range(len(csiq_video_list)):
#    name = csiq_video_list[i] 
#    scene = np.array(np.where(csiq_seq == name.split('_')[0])).squeeze()
#    idx = int(name.split('_')[-1])
#    
#    if int(idx) in awgn:
#        continue
#    get_video = skvideo.io.vread('/media/ece/DATA/Shankhanil/VQA/CSIQ/csiq_videos/' +
#                                       name + '.yuv', 480, 832, as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'})
#    
#    for j in range(0,len(get_video)-1,2):
#        frame_diff = get_video[j].astype(np.int16) - get_video[j+1].astype(np.int16)
#        
#        pred_temporal = temporal_model.predict(np.expand_dims(frame_diff,0)).squeeze()        
#        t_predict_score[count].append(pred_temporal.mean())
#        
#    count = count + 1
#    print(count)
#    
#pretrained_scores = np.load('/media/ece/DATA/Shankhanil/VQA/tmp/spatial_allvscsiq_resnet50.npy')
##pretrained_scores = np.load('/media/ece/DATA/Shankhanil/VQA/tmp/spatial_evvqvscsiq_resnet50.npy')
#x1 =[]
#trred = []
#for i in range(len(t_predict_score)):
#    x1.append(pretrained_scores[i] * np.mean(t_predict_score[i]))
#    trred.append(np.mean(t_predict_score[i]))
#
#test_idx  = np.array([18*i + np.concatenate([np.arange(12),  np.arange(15,18)]) for i in range(12)]).flatten()
#z_dmos,_ = srocc(x1, np.take(csiq_dmos,test_idx))   
#
#test_dmos = np.take(csiq_dmos,test_idx)
#srred = pretrained_scores
#trred = np.array(trred)
#strred = np.array(x1)
#
#mdic = {"srred":srred, "trred":trred, "strred":strred, "dmos":test_dmos}
#savemat("/media/ece/DATA/Shankhanil/VQA/tmp/allvscsiq.mat", mdic)
    
################### ECVQ ###########################################################
#temporal_model = load_model(allvs1_directory + 'allvsecvq/' + 'temporal_allvsecvq_19000.h5')
#ecvq_directory = '/media/ece/DATA/Shankhanil/VQA/ECVQ/cif_videos/'
#with open(ecvq_directory + 'subjective_scores_cif.txt') as f:
#    file_names = f.readlines()
#ecvq_video_list = [x.split()[0] for x in file_names] 
#ecvq_dmos_list  = [float(x.split()[2]) for x in file_names]
#ecvq_dmos = np.array(ecvq_dmos_list)
#
#ecvq_s_rred = sio.loadmat(strred_directory + 'strred_ecvq.mat')['srred']
#
#ecvq_seq = np.array(['container_ship', 'flower_garden', 'football', 'foreman', 'hall', 'mobile', 'news', 'silent'])
#  
#t_predict_score = []
#for i in range(len(ecvq_video_list)):
#    t_predict_score.append([])
#    
#for i in range(len(ecvq_video_list)):
#    name = ecvq_video_list[i] 
#    scene = np.array(np.where(ecvq_seq == name.rsplit('_', 1)[0])).squeeze()
#    idx = int(name.rsplit('_', 1)[-1])
#    
#    get_video = skvideo.io.vread('/media/ece/DATA/Shankhanil/VQA/ECVQ/cif_videos/' +
#                                       name + '.yuv', 288, 352, as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'})
#    
#    for j in range(0,298,2):
#        frame_diff = get_video[j].astype(np.int16) - get_video[j+1].astype(np.int16)
#        
#        pred_temporal = temporal_model.predict(np.expand_dims(frame_diff,0)).squeeze()        
#        t_predict_score[i].append(pred_temporal.mean())
#    print(i)
#    
#pretrained_scores = np.load('/media/ece/DATA/Shankhanil/VQA/tmp/spatial_allvsecvq_resnet50.npy')
##pretrained_scores = np.load('/media/ece/DATA/Shankhanil/VQA/tmp/spatial_evvqvsecvq_resnet50.npy')
#x1 =[]
#trred = []
#
#for i in range(len(t_predict_score)):
#    x1.append(pretrained_scores[i] * np.mean(t_predict_score[i]))
#    trred.append(np.mean(t_predict_score[i]))
#z_dmos,_ = srocc(x1, ecvq_dmos)
#  
#srred = pretrained_scores
#trred = np.array(trred)
#strred = np.array(x1)
#
#mdic = {"srred":srred, "trred":trred, "strred":strred, "dmos":ecvq_dmos}
#savemat("/media/ece/DATA/Shankhanil/VQA/tmp/allvsecvq.mat", mdic)
#################### EVVQ ###########################################################
#temporal_model = load_model(allvs1_directory + 'allvsevvq/' + 'temporal_allvsevvq_19000.h5')
#evvq_directory = '/media/ece/DATA/Shankhanil/VQA/EVVQ/vga_videos/'
#with open(evvq_directory + 'subjective_scores_vga.txt') as f:
#    file_names = f.readlines()
#evvq_video_list = [x.split()[0] for x in file_names] 
#evvq_dmos_list  = [float(x.split()[2]) for x in file_names]
#evvq_dmos = np.array(evvq_dmos_list)
#
#evvq_s_rred = sio.loadmat(strred_directory + 'strred_evvq.mat')['srred']
#
#evvq_seq = np.array(['cartoon', 'cheerleaders', 'discussion', 'flower_garden', 'football', 'mobile', 'town_plan', 'weather'])
#
#t_predict_score = []
#for i in range(len(evvq_video_list)):
#    t_predict_score.append([])
#    
#for i in range(len(evvq_video_list)):
#    name = evvq_video_list[i] 
#    scene = np.array(np.where(evvq_seq == name.rsplit('_', 1)[0])).squeeze()
#    idx = int(name.rsplit('_', 1)[-1])
#    
#    get_video = skvideo.io.vread('/media/ece/DATA/Shankhanil/VQA/EVVQ/vga_videos/' +
#                                       name + '.yuv', 480, 640, as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'})
#    
#    for j in range(0,298,2):
#        frame_diff = get_video[j].astype(np.int16) - get_video[j+1].astype(np.int16)
#        
#        pred_temporal = temporal_model.predict(np.expand_dims(frame_diff,0)).squeeze()        
#        t_predict_score[i].append(pred_temporal.mean())
#    print(i)
#pretrained_scores = np.load('/media/ece/DATA/Shankhanil/VQA/tmp/spatial_allvsevvq_resnet50.npy')
##pretrained_scores = np.load('/media/ece/DATA/Shankhanil/VQA/tmp/spatial_ecvqvsevvq_resnet50.npy')
#
#x1 =[]
#trred = []
#
#for i in range(len(t_predict_score)):
#    x1.append(pretrained_scores[i] * np.mean(t_predict_score[i]))
#    trred.append(np.mean(t_predict_score[i]))
#z_dmos,_ = srocc(x1, evvq_dmos)
#  
#srred = pretrained_scores
#trred = np.array(trred)
#strred = np.array(x1)
#
#mdic = {"srred":srred, "trred":trred, "strred":strred, "dmos":evvq_dmos}
#savemat("/media/ece/DATA/Shankhanil/VQA/tmp/allvsevvq.mat", mdic)


################################## KONVID ###########################################
temporal_model = load_model('/media/ece/DATA/Shankhanil/VQA/tmp/temporal_allvsmobile_30000.h5')
#temporal_model =load_model('/media/ece/DATA/Shankhanil/VQA/tmp/temporal_allvsmobile_shallownet.h5')
import csv

flickr_id  = []
konvid_mos = []
konvid_resnet = []
konvid_directory = '/media/ece/DATA/Shankhanil/VQA/konvid/KoNViD_1k_videos/' 

with open(konvid_directory + 'KoNViD_1k_mos.csv', 'r') as file:
    reader = csv.reader(file) 
    for row in reader:
        flickr_id.append(row[0])
        konvid_mos.append(row[1])
        
t_predict_score = []
for i in range(len(flickr_id)):
    t_predict_score.append([])
    
flickr_id  = flickr_id[1:]
konvid_mos = konvid_mos[1:]
konvid_dmos = [5-float(i) for i in konvid_mos]

for i in range(len(flickr_id)):
    name = flickr_id[i]
    get_video = skvideo.io.vread(konvid_directory +
                                       name + '.mp4', as_grey = True)
    
    for j in range(0,len(get_video)-1,2):
        frame_diff = get_video[j].astype(np.int16) - get_video[j+1].astype(np.int16)
        
        pred_temporal = temporal_model.predict(np.expand_dims(frame_diff,0)).squeeze()        
        t_predict_score[i].append(pred_temporal.mean())
    print(i)
#pretrained_scores = np.load('/media/ece/DATA/Shankhanil/VQA/tmp/spatial_allvskonvid_resnet50.npy')
pretrained_scores =loadmat('/media/ece/Shankhanil/TMM/nrsted/konvid_srred.mat')['srred'].squeeze()
x1 =[]
for i in range(1200):
    x1.append(np.mean(t_predict_score[i]))# * pretrained_scores[i] )
z_dmos,_ = srocc(x1, konvid_dmos)    
print(z_dmos)