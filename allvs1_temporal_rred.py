import tensorflow as tf
from keras.applications import VGG19
from keras.applications.resnet50 import ResNet50
from keras import models
from keras.layers  import Conv2D, Dense,Flatten,Dropout,GlobalMaxPooling2D, GlobalAveragePooling2D, Input,Lambda, BatchNormalization
from keras import optimizers, layers
from keras import backend as K
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.layers import GRU, LSTM, TimeDistributed,Layer,concatenate, PReLU, LeakyReLU, Concatenate
from tensorflow.keras.layers import Cropping2D
import glob
import cv2
from scipy import io as sio
import numpy as np
import os
#import matplotlib.pyplot as plt
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

from keras import regularizers
import h5py
from keras import losses

from scipy.io import loadmat
import skvideo.io

from sklearn import linear_model
from sklearn import preprocessing
from sklearn.kernel_approximation import Nystroem
from itertools import combinations
import time
import random
import gc
from keras.utils import multi_gpu_model
from matplotlib import pyplot as plt

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
gc.collect()
K.clear_session()

strred_directory = '/media/ece/DATA/Shankhanil/VQA/strred_files/'
##################################  LIVE VQA  #####################################

live_directory = '/media/ece/DATA/Shankhanil/VQA/live_vqa/'
with open(live_directory + 'live_video_quality_seqs.txt') as f:
    video_names = f.readlines()
live_video_list = [x.strip('.yuv\n') for x in video_names] 

with open(live_directory + 'live_video_quality_data.txt') as f:
    video_dmos = f.readlines()
live_dmos_list = [float(x.split('\t')[0]) for x in video_dmos] 
live_dmos = np.array(live_dmos_list)

#feat = np.zeros([len(video_list),2048*2])

live_seq = np.array(['pa', 'rb', 'rh', 'tr', 'st', 'sf', 'bs', 'sh', 'mc', 'pr'])
rate = [25,25,25,25,25,25,25,50,50,50]
live_trred = sio.loadmat(strred_directory + 'strred_live.mat')['trred']

seq_id = {0:'pa', 1:'rb', 2:'rh', 3:'tr', 4:'st', 5:'sf', 6:'bs', 7:'sh',  8:'mc', 9:'pr'}

########################## EPFL-POLIMI CIF #####################################

epfl_cif_directory = '/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/decoded/CIF/'
with open(epfl_cif_directory + 'names_scores.txt') as f:
    file_names = f.readlines()
epfl_cif_video_list = [x.split('\t')[0] for x in file_names] 
epfl_cif_dmos_list  = [float(x.split('\t')[1]) for x in file_names]
epfl_cif_dmos = 5-np.array(epfl_cif_dmos_list)

epfl_cif_trred = sio.loadmat(strred_directory + 'strred_epfl_cif.mat')['trred']

epfl_cif_seq = np.array(['foreman','hall','mobile','mother','news','paris'])

########################## EPFL-POLIMI 4CIF #####################################

epfl_4cif_directory = '/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/decoded/4CIF/'
with open(epfl_4cif_directory + 'names_scores.txt') as f:
    file_names = f.readlines()
epfl_4cif_video_list = [x.split('\t')[0] for x in file_names] 
epfl_4cif_dmos_list  = [float(x.split('\t')[1]) for x in file_names]
epfl_4cif_dmos = 5-np.array(epfl_4cif_dmos_list)

epfl_4cif_trred = sio.loadmat(strred_directory + 'strred_epfl_4cif.mat')['trred']

epfl_4cif_seq = np.array(['CROWDRUN','DUCKSTAKEOFF','HARBOUR','ICE','PARKJOY','SOCCER'])


############################### Mobile LIVE ###################################
mobile_directory = '/media/ece/DATA/Shankhanil/VQA/live_mobile/'
mobile_dmos = sio.loadmat(strred_directory + 'strred_mobile.mat')['dmos'].squeeze()

mobile_trred = sio.loadmat(strred_directory + 'strred_mobile.mat')['trred'].squeeze()
#mobile_t_rred = np.reshape(mobile_t_rred.copy(), (10,16))
mobile_video_list = sio.loadmat(strred_directory + 'strred_mobile.mat')['names'].squeeze()
#mobile_seq_id = {0:'bf', 1:'dv', 2:'fc', 3:'hc', 4:'la', 5:'po', 6:'rb', 7:'sd', 8:'ss', 9:'tk'}
mobile_seq = np.array(['bf', 'dv', 'fc', 'hc', 'la', 'po', 'rb', 'sd', 'ss', 'tk'])
mobile_dist = np.array(['r1','r2','r3','r4','s14','s24','s34','t14','t124','t421','t134','t431','w1','w2','w3','w4'])

###############################  CSIQ  #############################################
csiq_directory = '/media/ece/DATA/Shankhanil/VQA/CSIQ/csiq_videos/'
with open(csiq_directory + 'video_subj_ratings.txt') as f:
    file_names = f.readlines()
csiq_video_list = [x.split('\t')[0].split('.')[0] for x in file_names] 
csiq_dmos_list  = [float(x.split('\t')[1]) for x in file_names]
csiq_dmos = np.array(csiq_dmos_list)

csiq_trred = sio.loadmat(strred_directory + 'strred_csiq.mat')['trred']

csiq_seq = np.array(['BQMall', 'BQTerrace','BasketballDrive','Cactus', \
            'Carving','Chipmunks','Flowervase','Keiba','Kimono', \
            'ParkScene','PartyScene','Timelapse'])

################## ECVQ ###########################################################
ecvq_directory = '/media/ece/DATA/Shankhanil/VQA/ECVQ/cif_videos/'
with open(ecvq_directory + 'subjective_scores_cif.txt') as f:
    file_names = f.readlines()
ecvq_video_list = [x.split()[0] for x in file_names] 
ecvq_dmos_list  = [float(x.split()[2]) for x in file_names]
ecvq_dmos = np.array(ecvq_dmos_list)

ecvq_trred = sio.loadmat(strred_directory + 'strred_ecvq.mat')['trred']

ecvq_seq = np.array(['container_ship', 'flower_garden', 'football', 'foreman', 'hall', 'mobile', 'news', 'silent'])

################## EVVQ ###########################################################
evvq_directory = '/media/ece/DATA/Shankhanil/VQA/EVVQ/vga_videos/'
with open(evvq_directory + 'subjective_scores_vga.txt') as f:
    file_names = f.readlines()
evvq_video_list = [x.split()[0] for x in file_names] 
evvq_dmos_list  = [float(x.split()[2]) for x in file_names]
evvq_dmos = np.array(evvq_dmos_list)

evvq_trred = sio.loadmat(strred_directory + 'strred_evvq.mat')['trred']

evvq_seq = np.array(['cartoon', 'cheerleaders', 'discussion', 'flower_garden', 'football', 'mobile', 'town_plan', 'weather'])

################################################################################

live_frame_dir      = '/media/ece/DATA/Shankhanil/VQA/live_vqa/live_vqa_frames_diff_gray/distorted/'
epfl_cif_frame_dir  = '/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/epfl_cif_frames_diff_gray/distorted/'
epfl_4cif_frame_dir = '/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/epfl_4cif_frames_diff_gray/distorted/'
mobile_frame_dir    = '/media/ece/DATA/Shankhanil/VQA/live_mobile/live_mobile_frames_diff_gray/distorted/'
csiq_frame_dir      = '/media/ece/DATA/Shankhanil/VQA/CSIQ/csiq_frames_diff_gray/distorted/'
ecvq_frame_dir      = '/media/ece/DATA/Shankhanil/VQA/ECVQ/ecvq_frames_diff_gray/distorted/'
evvq_frame_dir      = '/media/ece/DATA/Shankhanil/VQA/EVVQ/evvq_frames_diff_gray/distorted/'


from natsort import natsort_keygen,natsorted, ns
natsort_key = natsort_keygen(alg=ns.IGNORECASE)

live_filenames = []
for ref in range(10):
    seq_path = live_frame_dir + live_seq[ref] + '/'
    live_filenames.append(natsorted([file for file in glob.glob(seq_path +'*.npy')], alg=ns.IGNORECASE))


epfl_cif_filenames = []
for ref in range(6):
    seq_path = epfl_cif_frame_dir + epfl_cif_seq[ref] + '/'
    epfl_cif_filenames.append(natsorted([file  for file in glob.glob(seq_path +'*.npy')], alg=ns.IGNORECASE))
   
epfl_4cif_filenames = []
for ref in range(6):
    seq_path = epfl_4cif_frame_dir + epfl_4cif_seq[ref] + '/'
    epfl_4cif_filenames.append(natsorted([file  for file in glob.glob(seq_path +'*.npy')], alg=ns.IGNORECASE))
    
mobile_filenames = []
for ref in range(10):
    seq_path = mobile_frame_dir + mobile_seq[ref] + '/'
    mobile_filenames.append(natsorted([file for file in glob.glob(seq_path +'*.npy')], alg=ns.IGNORECASE))

csiq_filenames = []
for ref in range(12):
    seq_path = csiq_frame_dir + csiq_seq[ref] + '/'
    csiq_filenames.append(natsorted([file for file in glob.glob(seq_path +'*.npy')], alg=ns.IGNORECASE))

ecvq_filenames = []
for ref in range(8):
    seq_path = ecvq_frame_dir + ecvq_seq[ref] + '/'
    ecvq_filenames.append(natsorted([file for file in glob.glob(seq_path +'*.npy')], alg=ns.IGNORECASE))

evvq_filenames = []
for ref in range(8):
    seq_path = evvq_frame_dir + evvq_seq[ref] + '/'
    evvq_filenames.append(natsorted([file for file in glob.glob(seq_path +'*.npy')], alg=ns.IGNORECASE))
     
################### Call Model #####################################################

#pwd = os.getcwd()
#os.chdir('/home/ece/Shankhanil/VQA/classification_models_master/')
#from classification_models.keras import Classifiers
#ResNet18, preprocess_input = Classifiers.get('resnet18')    
#os.chdir(pwd)

def get_model():
#    base_model = ResNet18(input_shape=(None,None,1), weights=None, include_top=False)
    base_model = ResNet50(include_top=False, weights=None, input_shape=(None,None,1), pooling='avg')    
   
#    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(base_model.output)
    x = Dense(64, activation='relu')(x)
    output = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=[base_model.input], outputs=[output])    
       
    return(model)

base_model = get_model()
model = multi_gpu_model(base_model, gpus=2)
model.compile(optimizer = Adam(1e-4), loss = losses.mean_absolute_error)
  
####################### CALL PRETRAINED MODEL ##################################
#custom_obj = {}
#custom_obj['tf'] = tf
    
#base_model = load_model('/media/ece/DATA/Shankhanil/VQA/linear_models/temporal_3dbvsepfl_64000.h5')
#model = multi_gpu_model(base_model,gpus=2)
#model.layers[-2].set_weights(base_model.get_weights())
##model.set_weights(base_model.get_weights())
#model.compile(optimizer=base_model.optimizer, loss = losses.mean_absolute_error)

################## Generate clone pretrained model ########################################

#base_model1 = load_model('/media/ece/DATA/Shankhanil/VQA/Allvs1_models/temporal/allvsepfl/temporal_allvsepfl_10000.h5')
#base_model = keras.models.clone_model(base_model1)
#model = multi_gpu_model(base_model,gpus=2)
#model.compile(optimizer = Adam(1e-4), loss = losses.mean_absolute_error)

######################### Start Training ####################################
databases = {  'live'      :{'num':150, 'ref':10, 'files' : live_filenames,         'dmos' : live_dmos}, \
               'mobile'    :{'num':160, 'ref':10, 'files' : mobile_filenames,       'dmos' : mobile_dmos}, \
               'epfl_cif'  :{'num':72,  'ref':6,  'files' : epfl_cif_filenames,     'dmos' : epfl_cif_dmos},   \
               'epfl_4cif' :{'num':72,  'ref':6,  'files' : epfl_4cif_filenames,    'dmos' : epfl_4cif_dmos},   \
               'csiq'      :{'num':216, 'ref':12, 'files' : csiq_filenames,         'dmos' : csiq_dmos}, \
               'ecvq'      :{'num':90,  'ref':8,  'files' : ecvq_filenames,         'dmos' : ecvq_dmos},   \
               'evvq'      :{'num':90,  'ref':8,  'files' : evvq_filenames,         'dmos' : evvq_dmos} }

ecvq_num_seq = np.cumsum([0, 12, 11, 10, 11, 12, 10, 12, 12])
evvq_num_seq = np.cumsum([0, 12, 9, 12, 11, 11, 11, 12, 12])

start_time = time.time()
batch_size = 8
epoch = 1

corr_dmos = []
all_train_dbs = ['ecvq', 'evvq', 'live', 'epfl_cif', 'epfl_4cif', 'csiq']#['live', 'mobile', 'csiq', 'ecvq', 'evvq']
all_test_dbs = ['mobile']
#corr_dmos = {'epfl_cif':0, 'epfl_4cif':0}

while(1):
#    break
    X_train = []
    y_train = []
    
    train_db = all_train_dbs[np.mod(epoch, len(all_train_dbs))]
    idx1 = np.random.choice(databases[train_db]['ref'], batch_size, True)
    filenames = databases[train_db]['files']
        
    for i in idx1: 
        names = filenames[i]        
        frame_id = np.random.randint(len(names))        
               
        dist_name = names[frame_id]
        dist_img = np.load(dist_name) 
        
        if train_db == 'live':
            x,y = dist_name.split('/')[-1].split('_')            ## LIVE TEST SETTING
            rred = live_trred[i][int(x[2:]) -1][0,int(y.split('.')[0])//2] 
        
        elif train_db == 'epfl_4cif':
            _,x,y = dist_name.split('/')[-1].split('_')         ## EPFL 4CIF TEST SETTING
            rred = epfl_4cif_trred[i][int(x)][0,int(y.split('.')[0])//2]
        
        elif train_db == 'epfl_cif':
            _,x,y = dist_name.split('/')[-1].split('_')         ## EPFL CIF TEST SETTING 
            rred = epfl_cif_trred[i][int(x)][0,int(y.split('.')[0])//2]
        
        elif train_db == 'mobile':                                   
            _,x,y = dist_name.split('/')[-1].split('_')        ## MOBILE TEST SETTING 
            rred = mobile_trred[16*i + int(np.where(mobile_dist==x)[0])][0,int(y.split('.')[0])//2]
        
        elif train_db == 'csiq':
            _,x,y = dist_name.split('/')[-1].split('_')         ## CSIQ TEST SETTING
            rred = csiq_trred[i][int(x) - 1][0,int(y.split('.')[0])//2]
        
        elif train_db == 'ecvq':
            _,x,y = dist_name.split('/')[-1].rsplit('_',2)          ## ECVQ TEST SETTING
            rred = ecvq_trred[i][int(x) - 1][0,int(y.split('.')[0])//2]
        
        elif train_db == 'evvq':
            _,x,y = dist_name.split('/')[-1].rsplit('_',2)          ## EVVQ TEST SETTING
            rred = evvq_trred[i][int(x) - 1][0,int(y.split('.')[0])//2]
                         
        if np.isfinite(rred)==False: 
            continue
        
        X_train.append(dist_img)
        y_train.append(np.expand_dims(rred,-1)/100)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
       
#    current_lr = np.piecewise(float(epoch), [epoch<10000, epoch>=10000], [5e-4,1e-4])
#    K.set_value(model.optimizer.lr, current_lr) 
    
    model.train_on_batch(X_train, y_train)
#    break
    if epoch and epoch%1000 == 0:
        
        x1 = []
        test_dmos = []
        for test_db in all_test_dbs:
            true_score = []
            predict_score = []
            
            idx2 = tuple(range(databases[test_db]['ref']))
            filenames = databases[test_db]['files']
            tmp = databases[test_db]['dmos']
            test_dmos.extend(tmp)
            
            for i in range(databases[test_db]['num']):            
                true_score.append([])
                predict_score.append([])
                
            count = 0
            
            for i in idx2:
            
                names = filenames[i]               
                test_id = np.random.choice(np.arange(len(names)),len(names)//5, False)
                
                for frame_id in test_id:
                                    
                    dist_name = names[frame_id] 
                    dist_img   = np.load(dist_name)
                    
                    if test_db == 'live':
                        x,y = dist_name.split('/')[-1].split('_')            ## LIVE TEST SETTING
                        trred = live_trred[i][int(x[2:]) -1][0,int(y.split('.')[0])//2] 
                        y_predict = base_model.predict(np.expand_dims(dist_img,0)).squeeze()                 
                        predict_score[15*count + int(names[frame_id].split('/')[-1].split('_')[0][2:]) - 2].append(np.mean(y_predict))
                    
                    elif test_db == 'epfl_4cif':
                        _,x,y = dist_name.split('/')[-1].split('_')         ## EPFL 4CIF TEST SETTING
                        trred = epfl_4cif_trred[i][int(x)][0,int(y.split('.')[0])//2]
                        y_predict = base_model.predict(np.expand_dims(dist_img,0)).squeeze()
                        predict_score[12*count + int(names[frame_id].split('/')[-1].split('_')[1])].append(np.mean(y_predict))
                        
                    elif test_db == 'epfl_cif':
                        _,x,y = dist_name.split('/')[-1].split('_')         ## EPFL CIF TEST SETTING 
                        trred = epfl_cif_trred[i][int(x)][0,int(y.split('.')[0])//2]
                        y_predict = base_model.predict(np.expand_dims(dist_img,0)).squeeze()
                        predict_score[12*count + int(names[frame_id].split('/')[-1].split('_')[1])].append(np.mean(y_predict))
                        
                    elif test_db == 'mobile':                                   
                        _,x,y = dist_name.split('/')[-1].split('_')        ## MOBILE TEST SETTING 
                        trred = mobile_trred[16*i + int(np.where(mobile_dist==x)[0])][0,int(y.split('.')[0])//2]
                        y_predict = base_model.predict(np.expand_dims(dist_img,0)).squeeze()
                        predict_score[16*count + int(np.where(mobile_dist==x)[0])].append(np.mean(y_predict))
                    
                    elif test_db == 'csiq':
                        _,x,y = dist_name.split('/')[-1].split('_')         ## CSIQ TEST SETTING
                        trred = csiq_trred[i][int(x) - 1][0,int(y.split('.')[0])//2]
                        y_predict = base_model.predict(np.expand_dims(dist_img,0)).squeeze()
                        predict_score[18*count + int(names[frame_id].split('/')[-1].split('_')[1]) - 1].append(np.mean(y_predict))
    
                    elif test_db == 'ecvq':
                        _,x,y = dist_name.split('/')[-1].rsplit('_',2)          ## ECVQ TEST SETTING
                        trred = ecvq_trred[i][int(x) - 1][0,int(y.split('.')[0])//2]
                        y_predict = base_model.predict(np.expand_dims(dist_img,0)).squeeze()
                        predict_score[ecvq_num_seq[count] + int(names[frame_id].split('/')[-1].rsplit('_',2)[1]) - 1].append(np.mean(y_predict))
                    
                    elif test_db == 'evvq':
                        _,x,y = dist_name.split('/')[-1].rsplit('_',2)          ## EVVQ TEST SETTING
                        trred = evvq_trred[i][int(x) - 1][0,int(y.split('.')[0])//2]
                        y_predict = base_model.predict(np.expand_dims(dist_img,0)).squeeze()
                        predict_score[evvq_num_seq[count] + int(names[frame_id].split('/')[-1].rsplit('_',2)[1]) - 1].append(np.mean(y_predict))               
                    
                    if np.isfinite(rred)==False: 
                        continue             
            
                count = count + 1         
            
            for i in range(len(predict_score)):               
                x1.append(np.mean(predict_score[i]))
        
        if test_db == 'csiq':
            test_idx  = np.array([18*i + np.concatenate([np.arange(12),  np.arange(15,18)]) for i in idx2]).flatten() 
            z_dmos,_ = srocc(np.take(x1,test_idx),np.take(test_dmos,test_idx))
        
        else:
            z_dmos,_ = srocc(x1, test_dmos)
                    
        corr_dmos.append(z_dmos)
        
        duration = (time.time() - start_time)/60        
#        print('epoch = %4d , corr_cif = %4.6f, corr_4cif = %4.6f, time = %4.2f m' %(epoch, corr_dmos['epfl_cif'], corr_dmos['epfl_4cif'], duration))
        print('epoch = %4d , corr_dmos = %4.6f, time = %4.2f m' %(epoch, z_dmos, duration))
#        if epoch and epoch %10000 ==0:
#           break
        output_directory = '/media/ece/DATA/Shankhanil/VQA/'
#        np.save(output_directory + 'tmp/' + 'mos_curve_csiq', corer herer_dmos)
        if  epoch and epoch%1000 ==0:
            base_model.compile(optimizer=model.optimizer, loss = losses.mean_absolute_error)
#            np.save(output_directory + 'linear_models/dmos_curve', corr_dmos)
            if epoch>=30000:
                base_model.save(output_directory + 'tmp/' +'temporal_allvsmobile_3fcmodel_' + str(epoch) + '.h5')
                break
            else:
                base_model.save(output_directory + 'tmp/' +'temporal_3dbvsmobile_3fcmodel_inter' + '.h5')
            
    epoch = epoch + 1
    gc.collect()
    