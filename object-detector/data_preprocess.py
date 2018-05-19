# coding=utf-8
import os
import numpy as np
from glob import glob
from PIL import Image
import random
from matplotlib import pyplot as plt

data_dir='/media/dilligencer/A67A6AF37A6ABFA3/DATASET_BREAST/3D_mask/test/'
label_dir='/media/dilligencer/A67A6AF37A6ABFA3/DATASET_BREAST/2d_preprocessed/'
pos_image_path='/media/dilligencer/A67A6AF37A6ABFA3/DATASET_BREAST/SVM-data/test/pos/'
neg_image_path='/media/dilligencer/A67A6AF37A6ABFA3/DATASET_BREAST/SVM-data/test/neg/'


file_list=glob(data_dir+'*clean.npy')
_pos_num=0
_neg_num=0
for i in file_list:
    print(i)
    image_name=i.split('/')[-1].split('-')[0]
    # print(image_name)
    label_list=glob(label_dir+image_name+'*.txt')
    # print(label_list)
    slice_ind_list=[]
    dcm_series=np.squeeze(np.load(i))
    # print(len(dcm_series))
    for j in label_list:
        _pos_num += 1
        slice_ind=int(j.split('-')[-1].replace('.txt',''))
        slice_ind_list.append(slice_ind)
        dcm=dcm_series[slice_ind][:300,:]
        image=Image.fromarray(dcm.astype(np.uint8))
        save_path=pos_image_path+'pos-'+str(_pos_num)+'.pgm'
        image.save(save_path)
    neg_list=[ind for ind in range(len(dcm_series)) if ind not in slice_ind_list]
    if len(slice_ind_list) < len(neg_list):
        neg_slice_list=random.sample(neg_list,len(slice_ind_list))
    else:
        neg_slice_list=neg_list
    for k in neg_slice_list:
        _neg_num += 1
        dcm=dcm_series[k][:300,:]
        image = Image.fromarray(dcm.astype(np.uint8))
        save_path = neg_image_path + 'neg-' + str(_neg_num) + '.pgm'
        image.save(save_path)












