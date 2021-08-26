# NR-STED
## Description:

This work implements “Predicting Spatio-Temporal Entropic Differences for
Robust No Reference Video Quality Assessment” in keras/tensorflow. If you are using the codes, cite the following article:

>S. Mitra, R. Soundararajan, and S. S. Channappayya, “Predicting Spatio-Temporal Entropic Differences for Robust No Reference Video Quality Assessment,” IEEE Signal Processing Letters, vol. 28, pp. 170–174, 2021.
 
![strred_img](https://user-images.githubusercontent.com/35575223/130897322-aa60817a-0134-4f40-82f4-e0b9bc373256.png)

## **ResNet Feature Extraction**:
We use the function feature_generate_resnet50.py to produce spatially aware quality feature using pre-trained ResNet50 keras weights.

## **Frame and Frame Difference Extraction**: 
Use the file vid2frame&framediff.py to generate frame differences from videos which will be used as input to temporal learning model.

## **Spatial_RRED Predictor**:
Use the allvs1_spatial_rred.py file to train and predict our spatial RRED model on any number of datasets. The following code requires ResNet50 feature for the video frames as input.

## **Temporal_RRED_model**: 
allvs1_temporal_rred.py train the temporal learning model from scrath for given training databases.

## **Prediction**:
Overall NR-STED index of videos are predicted using test_st_rredmap_framelvl.py. Where trained temporal_rred model and spatial_rred prediction are taken as input.

## ** Pre-trained Spatial Model**:
[Drive link](https://drive.google.com/file/d/1nssTD1MwD-C9JZavjfZ68fDhqXvKqP66/view?usp=sharing)
