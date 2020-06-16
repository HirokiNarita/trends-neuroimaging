import pandas as pd
import numpy as np
import h5py
import os
import time
import gc
import sys

from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
# scipy sparse
from scipy import sparse

# setting param
input_base_path = "/media/hiroki/share/kaggle_data/trends-assessment-prediction"
out_base_path = "/media/hiroki/working/kaggle_data/trends-neuroimaging/split_IC"
train_list = os.listdir(path=input_base_path+'/fMRI_train')
test_list = os.listdir(path=input_base_path+'/fMRI_test')
train_num_records = len(train_list)
test_num_records = len(test_list)
xyz = 52*63*53

def wrap_make_vec(args):
    return make_vec(*args)
# count_nonzero 
def make_vec(file_name, num_ic):
    #for row, file_name in enumerate(train_list):
    f = h5py.File(input_base_path+'/fMRI_train/'+file_name,'r')
    vec = (f['SM_feature'][:,:,:,num_ic].reshape(xyz)).tolist()
    del f
    gc.collect()
    return vec
    #if row == 0:
    #    coo_matrix = coo_vec
    #else:
    #    coo_matrix = sparse.vstack([coo_matrix, coo_vec])

def multi_ok(job_args1):
    with mp.get_context('spawn').Pool() as p:
        mtx = np.array(list(tqdm(p.imap(wrap_make_vec, job_args1, chunksize=20), total=len(train_list))))
    sparse.save_npz(out_base_path+"/train/ic{}_matrix.npz".format(num_ic+1), sparse.csr_matrix(mtx))
    gc.collect()

if __name__== "__main__":
    print(sys.argv[1])
    num_ic = int(sys.argv[1])
    job_args1 = [(file_name, num_ic) for file_name in train_list]
    multi_ok(job_args1)