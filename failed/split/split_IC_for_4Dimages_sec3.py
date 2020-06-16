import pandas as pd
import numpy as np
import h5py
import os
import time
import gc

from tqdm import tqdm
from multiprocessing import Pool, cpu_count
# scipy sparse
from scipy import sparse

# setting param
input_base_path = "/media/hiroki/share/kaggle_data/trends-assessment-prediction"
out_base_path = "/media/hiroki/working/kaggle_data/trends-neuroimaging/split_IC"
train_list = os.listdir(path=input_base_path+'/fMRI_train')
#test_list = os.listdir(path=input_base_path+'/fMRI_test')
train_num_records = len(train_list)
#test_num_records = len(test_list)
xyz = 52*63*53


def io_num_component(num_ic):
    lil_matrix = sparse.lil_matrix((train_num_records, xyz))
    for row, file_name in enumerate(tqdm(train_list)):
        f = h5py.File(input_base_path+'/fMRI_train/'+file_name,'r')
        data = f['SM_feature']
        lil_vec = sparse.lil_matrix(data[:,:,:,num_ic].reshape(xyz))
        lil_matrix[row,:] = lil_vec
    sparse.save_npz(out_base_path+"/train/ic{}_matrix.npz".format(num_ic+1), lil_matrix.tocsr())
job_arg = [num_ic for num_ic in range(8,54)]

for num_ic in range(30,40):
    io_num_component(num_ic)
