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
from scipy.stats import spearmanr,pearsonr 
import keras
from keras.optimizers import Adam, Nadam
from keras.models import load_model
#layer_index = -1
#img_shape = (500,500,3)
from keras import regularizers
import h5py
from keras import losses

from scipy.io import loadmat

from sklearn import linear_model
from sklearn import preprocessing
from sklearn.kernel_approximation import Nystroem
from itertools import combinations
import time
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
K.clear_session()


#########################################################################################################    
spatial_feat_dir  = '../resnet50_feat/'
strred_directory = '../strred_files/'
all_train_dbs = ['epfl_4cif', 'epfl_cif','live', 'ecvq', 'csiq', 'evvq']
all_test_dbs = ['mobile']

##################################  LIVE VQA  #####################################
live_srred = []
live_resnet = []
live_ssim = []
live_mse = []

live_directory = '/media/ece/DATA/Shankhanil/VQA/live_vqa/'
with open(live_directory + 'live_video_quality_seqs.txt') as f:
    video_names = f.readlines()
live_video_list = [x.strip('.yuv\n') for x in video_names] 

with open(live_directory + 'live_video_quality_data.txt') as f:
    video_dmos = f.readlines()
live_dmos_list = [float(x.split('\t')[0]) for x in video_dmos] 
live_dmos = np.array(live_dmos_list)

seq = np.array(['pa', 'rb', 'rh', 'tr', 'st', 'sf', 'bs', 'sh', 'mc', 'pr'])
rate = [25,25,25,25,25,25,25,50,50,50]
live_s_rred   = sio.loadmat(strred_directory + 'strred_live.mat')['srred']

for i in range(len(live_video_list)):

    name = live_video_list[i]
    scene = np.array(np.where(seq == name[:2])).squeeze()
    idx = int(name.split('_')[0][2:])
 
    live_srred.append(live_s_rred[scene][idx - 1].squeeze())
    live_resnet.append(np.load(spatial_feat_dir + 'live_vqa/' + name + '.npy'))

######################### EPFL-4cif #####################################
epfl_4cif_srred = []
epfl_4cif_resnet = []
epfl_4cif_ssim = []
epfl_4cif_mse = []


epfl_4cif_directory = '/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/decoded/4CIF/'
with open(epfl_4cif_directory + 'names_scores.txt') as f:
    file_names = f.readlines()
epfl_4cif_video_list = [x.split()[0] for x in file_names] 
epfl_4cif_dmos_list  = [float(x.split()[1]) for x in file_names]
epfl_4cif_dmos = 5-np.array(epfl_4cif_dmos_list)

epfl_4cif_s_rred = sio.loadmat(strred_directory + 'strred_epfl_4cif.mat')['srred']


epfl_4cif_seq = np.array(['CROWDRUN','DUCKSTAKEOFF','HARBOUR','ICE','PARKJOY','SOCCER'])

for i in range(len(epfl_4cif_video_list)):
    
    name = epfl_4cif_video_list[i]
    scene = np.array(np.where(epfl_4cif_seq == name.split('_')[0])).squeeze()
    idx = np.mod(i,12)
    
    epfl_4cif_srred.append(epfl_4cif_s_rred[scene][idx].squeeze())
    
    epfl_4cif_resnet.append(np.load(spatial_feat_dir +  'epfl_4cif/' + name + '.npy'))

########################## EPFL-cif #####################################
epfl_cif_srred = []
epfl_cif_resnet = []
epfl_cif_ssim = []
epfl_cif_mse = []

epfl_cif_directory = '/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/decoded/CIF/'
with open(epfl_cif_directory + 'names_scores.txt') as f:
    file_names = f.readlines()
epfl_cif_video_list = [x.split()[0] for x in file_names] 
epfl_cif_dmos_list  = [float(x.split()[1]) for x in file_names]
epfl_cif_dmos = 5-np.array(epfl_cif_dmos_list)

epfl_cif_s_rred = sio.loadmat(strred_directory + 'strred_epfl_cif.mat')['srred']

epfl_cif_seq = np.array(['foreman','hall','mobile','mother','news','paris'])

for i in range(len(epfl_cif_video_list)):
    
    name = epfl_cif_video_list[i]
    scene = np.array(np.where(epfl_cif_seq == name.split('_')[0])).squeeze()
    idx = np.mod(i,12)
    
    epfl_cif_srred.append(epfl_cif_s_rred[scene][idx].squeeze())

    epfl_cif_resnet.append(np.load(spatial_feat_dir + 'epfl_cif/' + name + '.npy'))
    
############################### MOBILE  LIVE ######################################################

mobile_srred = []
mobile_resnet = []
mobile_ssim = []
mobile_mse = []

mobile_directory = '/media/ece/DATA/Shankhanil/VQA/live_mobile/'
mobile_dmos = sio.loadmat(strred_directory + 'dmos_live_mobile.mat')['dmos'].squeeze()

mobile_s_rred = sio.loadmat(strred_directory + 'strred_mobile.mat')['srred'].squeeze()

mobile_video_list = sio.loadmat(strred_directory + 'strred_mobile.mat')['names'].squeeze()
mobile_seq_id = {0:'bf', 1:'dv', 2:'fc', 3:'hc', 4:'la', 5:'po', 6:'rb', 7:'sd', 8:'ss', 9:'tk'}

for i in range(len(mobile_video_list)):
    
    name = str(mobile_video_list[i][0]) 
    
    mobile_srred.append(mobile_s_rred[i].squeeze())
    mobile_resnet.append(np.load(spatial_feat_dir + 'live_mobile/' + name + '.npy'))
    
###############################  CSIQ  #############################################
    
csiq_srred = []
csiq_resnet = []
csiq_ssim = []
csiq_mse = []

csiq_directory = '/media/ece/DATA/Shankhanil/VQA/CSIQ/csiq_videos/'
with open(csiq_directory + 'video_subj_ratings.txt') as f:
    file_names = f.readlines()
csiq_video_list = [x.split('\t')[0].split('.')[0] for x in file_names] 
csiq_dmos_list  = [float(x.split('\t')[1]) for x in file_names]
csiq_dmos = np.array(csiq_dmos_list)

csiq_s_rred = sio.loadmat(strred_directory + 'strred_csiq.mat')['srred']
csiq_seq = np.array(['BQMall', 'BQTerrace','BasketballDrive','Cactus', \
            'Carving','Chipmunks','Flowervase','Keiba','Kimono', \
            'ParkScene','PartyScene','Timelapse'])

awgn = [13,14,15]
for i in range(len(csiq_video_list)):
    name = csiq_video_list[i] 
    scene = np.array(np.where(csiq_seq == name.split('_')[0])).squeeze()
    idx = int(name.split('_')[-1])
    
    if int(idx) in awgn:
        continue
    csiq_srred.append(csiq_s_rred[scene][idx - 1].squeeze())

    csiq_resnet.append(np.load(spatial_feat_dir + 'csiq/' + name + '.npy'))
    
################## ECVQ ###########################################################
ecvq_srred = []
ecvq_resnet = []    
ecvq_ssim = []
ecvq_mse = []

ecvq_directory = '/media/ece/DATA/Shankhanil/VQA/ECVQ/cif_videos/'
with open(ecvq_directory + 'subjective_scores_cif.txt') as f:
    file_names = f.readlines()
ecvq_video_list = [x.split()[0] for x in file_names] 
ecvq_dmos_list  = [float(x.split()[2]) for x in file_names]
ecvq_dmos = np.array(ecvq_dmos_list)

ecvq_s_rred = sio.loadmat(strred_directory + 'strred_ecvq.mat')['srred']
ecvq_seq = np.array(['container_ship', 'flower_garden', 'football', 'foreman', 'hall', 'mobile', 'news', 'silent'])
    
for i in range(len(ecvq_video_list)):
    name = ecvq_video_list[i] 
    scene = np.array(np.where(ecvq_seq == name.rsplit('_', 1)[0])).squeeze()
    idx = int(name.rsplit('_', 1)[-1])
    
    ecvq_srred.append(ecvq_s_rred[scene][idx - 1].squeeze())
    
    ecvq_resnet.append(np.load(spatial_feat_dir + 'ecvq/' + name + '.npy'))   
#    
################### EVVQ ###########################################################
evvq_srred = []
evvq_resnet = []
evvq_ssim = []
evvq_mse = []

evvq_directory = '/media/ece/DATA/Shankhanil/VQA/EVVQ/vga_videos/'
with open(evvq_directory + 'subjective_scores_vga.txt') as f:
    file_names = f.readlines()
evvq_video_list = [x.split()[0] for x in file_names] 
evvq_dmos_list  = [float(x.split()[2]) for x in file_names]
evvq_dmos = np.array(evvq_dmos_list)

evvq_s_rred = sio.loadmat(strred_directory + 'strred_evvq.mat')['srred']

evvq_seq = np.array(['cartoon', 'cheerleaders', 'discussion', 'flower_garden', 'football', 'mobile', 'town_plan', 'weather'])

for i in range(len(evvq_video_list)):
    name = evvq_video_list[i] 
    scene = np.array(np.where(evvq_seq == name.rsplit('_', 1)[0])).squeeze()
    idx = int(name.rsplit('_', 1)[-1])
    
    evvq_srred.append(evvq_s_rred[scene][idx - 1].squeeze())
    
    evvq_resnet.append(np.load(spatial_feat_dir + 'evvq/' + name + '.npy'))   


#################################### Network ####################################################
def fine_tune_model(input_shape = (2048,)):
    
#    distorted_input = Input(input_shape)

    model = Sequential()
    model.add(Dense(1024, activation = 'relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0e-3),input_shape=input_shape))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation = 'relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0e-3)))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation = 'relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0e-3)))
    model.add(Dropout(0.4))
#    model.add(Lambda(lambda  x: K.l2_normalize(x,axis=-1)))
    model.add(Dense(1, activation = 'relu'))
    
    model.compile(optimizer=Adam(lr = 1e-5),
                  loss=losses.mean_squared_error)    
    return(model)

#########################################################################################################
    
databases = {  'live'      :{'num':150, 'ref':10, 'srred' : live_srred,      'feat' : live_resnet,       'dmos' : live_dmos}, \
               'mobile'    :{'num':160, 'ref':10, 'srred' : mobile_srred,   'feat' : mobile_resnet,     'dmos' : mobile_dmos}, \
               'epfl_cif'  :{'num':72,  'ref':6,  'srred' : epfl_cif_srred, 'feat' : epfl_cif_resnet,   'dmos' : epfl_cif_dmos},   \
               'epfl_4cif' :{'num':72,  'ref':6,  'srred' : epfl_4cif_srred,'feat' : epfl_4cif_resnet,  'dmos' : epfl_4cif_dmos},   \
               'csiq'      :{'num':216, 'ref':12, 'srred' : csiq_srred,    'feat' : csiq_resnet,       'dmos' : csiq_dmos}, \
               'ecvq'      :{'num':90,  'ref':8,  'srred' : ecvq_srred,    'feat' : ecvq_resnet,       'dmos' : ecvq_dmos},   \
               'evvq'      :{'num':90,  'ref':8,  'srred' : evvq_srred,     'feat' : evvq_resnet,       'dmos' : evvq_dmos}}

gt = 'srred'

rho_test =[]
rho_train = []
rho_srred = []

start_time = time.time()
#combs = list(combinations(np.arange(6),0))

train_data   = []
train_target = []

for i in range(len(all_train_dbs)):
    train_data.extend(databases[all_train_dbs[i]]['feat'])
    train_target.extend(databases[all_train_dbs[i]][gt])


X_train = train_data[0]
y_train = train_target[0]
for i in range(1,len(train_data)):
    X_train = np.concatenate((X_train, train_data[i]), axis = 0)
    y_train = np.concatenate((y_train, train_target[i]), axis = 0)
y_train = y_train#/100.0

dims = np.where(np.isfinite(y_train))
X_train = np.take(X_train, dims, 0).squeeze()
y_train = np.take(y_train, dims).squeeze()

model = fine_tune_model()
model.fit(X_train, y_train, validation_split=0.0,
              batch_size = 16, epochs = 20, verbose=0)

#scaler = preprocessing.MaxAbsScaler().fit(X_train)
#model = SVR(gamma= 'scale', C= 1.0, epsilon=0.2)
#clf = LinearSVR(random_state=0, tol=1e-5, max_iter=5000)
#X_scaler = scaler.transform(X_train)
#
#    X_scaler = np.take(X_scaler, np.where(y_train!=0)[0], axis =0)
#    y_train = np.take(y_train, np.where(y_train!=0)[0])
#
#rnd_idx = np.random.choice(len(y_train), 20000, replace = False)
#model.fit(np.take(X_train, rnd_idx,0), np.take(y_train, rnd_idx))

for test_db in all_test_dbs:
    test_data   = []
    test_target = []
    test_dmos   = []
    
    for i in range(len(all_test_dbs)):
        test_data.extend(databases[test_db]['feat'])
        test_target.extend(databases[test_db][gt])
        test_dmos.extend(databases[test_db]['dmos'])
    
    s_predict_test = []
    s_true_test = []
    srred_test = []
    
    for i in range(len(test_data)):
       dims = np.where(np.isfinite(test_target[i]))
       X_tmp = np.take(test_data[i], dims,0).squeeze()
       y_tmp = np.take(test_target[i], dims).squeeze()
       
    #       s_predict_test.append(model.predict(scaler.transform(X_tmp)))
       s_predict_test.append(model.predict(X_tmp))
       s_true_test.append(y_tmp)
       srred_test.append(np.mean(y_tmp))
    
       
    s_predict_qa = [np.mean(qa) for qa in s_predict_test]
    
    if all_test_dbs[0] == 'csiq':
        test_idx  = np.array([18*i + np.concatenate([np.arange(12),  np.arange(15,18)]) for i in range(12)]).flatten() 

        srocc,_ = spearmanr(np.take(test_dmos,test_idx), s_predict_qa)
        plcc,_ =  pearsonr(np.take(test_dmos,test_idx), s_predict_qa)
    else:
        
        srocc,_ = spearmanr(s_predict_qa, test_dmos)
        plcc,_ = pearsonr(s_predict_qa, test_dmos)

K.clear_session()
duration = (time.time() - start_time)/60
print(' spearman_corr = %4.6f,  pearson_corr = %4.6f, time = %4.2f m' %(srocc, plcc,duration))


output_dir= ''
np.save(output_dir+'spatial_allvsepfl_resnet50', np.array(s_predict_qa))