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
#list_path = "/media/hiroki/working/kaggle_data/trends-neuroimaging/split_IC/test/vecs/1"
out_base_path = "/media/hiroki/working/kaggle_data/trends-neuroimaging/split_IC"
train_list = os.listdir(path=input_base_path+'/fMRI_train')
test_list = os.listdir(path=input_base_path+'/fMRI_test')
train_num_records = len(train_list)
test_num_records = len(test_list)
xyz = 52*63*53

def wrap_make_vec(args):
    return load_vec_and_tolil(*args)
# count_nonzero

def load_vec_and_tolil(num_ic, file_name):
    f = h5py.File(input_base_path+'/fMRI_train/'+file_name,'r')
    data = f['SM_feature']
    np_array4D = data[...]
    # vectorize
    vec = np_array4D[num_ic,:,:,:].reshape(xyz)
    lil_vec = sparse.lil_matrix(vec)
    return lil_vec

def multi_ok(job_args1, num_ic):
    with mp.Pool() as p:
        vecs = list(tqdm(p.imap(wrap_make_vec, job_args1, chunksize=1), total=len(train_list)))
    lil_mtx = sparse.lil_matrix((train_num_records,xyz))
    for (row,vec) in enumerate(tqdm(vecs)):
        lil_mtx[row,:] = vec
        vecs[row] = []
    sparse.save_npz(out_base_path+"/train/IC/ic{}_matrix.npz".format(num_ic+1), lil_mtx.tocsr())
    gc.collect()
    p.close()
    p.join()


if __name__== "__main__":
    num_ic = int(sys.argv[1])
    job_args1 = [(num_ic, file_name) for file_name in train_list]
    multi_ok(job_args1, num_ic)
