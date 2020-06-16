import pandas as pd
import numpy as np
import h5py
import os
import time

import gc

# scipy sparse
from scipy import sparse
from scipy.sparse import save_npz
# multiprocessing
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from glob import glob

from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from tqdm import tqdm

class config:
    input_base_path = "/media/hiroki/share/kaggle_data/trends-assessment-prediction"
    out_base_path = "/media/hiroki/working/kaggle_data/trends-neuroimaging/split_IC"
    train_list = os.listdir(path=input_base_path+'/fMRI_train')
    test_list = os.listdir(path=input_base_path+'/fMRI_test')
    train_num_records = len(train_list)
    test_num_records = len(test_list)
    xyz = 52*63*53

def load_ic(file_name, num_ic):
    # load .mat
    f = h5py.File(config.input_base_path+'/fMRI_train/'+file_name,'r')
    data = f['SM_feature']
    np_array4D = data[:,:,:,num_ic]
    # vectorize
    num_ic_vec = lil_matrix(np_array4D.reshape(config.xyz))
    gc.collect()
    return num_ic_vec

for num_ic in range(40,54):
    
    start_time = time.time()
    # init group_num_ic matrix(train)
    train_num_ic_matrix = coo_matrix((config.train_num_records, config.xyz), dtype=np.float64).tolil()
    
    for (row,file_name) in enumerate(tqdm(config.train_list)):
        train_num_ic_matrix[row,:] = load_ic(file_name, num_ic)
        gc.collect()
    sparse.save_npz(config.out_base_path+"/train/ic{}_matrix.npz".format(num_ic+1), train_num_ic_matrix.tocsr())
    
    print("success : IC{}".format(num_ic+1))
    end_time = time.time()
    elapsed_time = end_time-start_time
    print("IC/sec:{}".format(elapsed_time))
    
    gc.collect()
