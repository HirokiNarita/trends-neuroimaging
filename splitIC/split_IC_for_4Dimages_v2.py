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
input_base_path = "/media/hiroki/working/kaggle_data/trends-neuroimaging/split_IC/train/vecs/"
train_list_path = "/media/hiroki/working/kaggle_data/trends-neuroimaging/split_IC/train/vecs/1"
test_list_path =  "/media/hiroki/working/kaggle_data/trends-neuroimaging/split_IC/test/vecs/4"
out_base_path = "/media/hiroki/working/kaggle_data/trends-neuroimaging/split_IC"
train_list = os.listdir(path=train_list_path)
test_list = os.listdir(path=test_list_path)
train_num_records = len(train_list)
test_num_records = len(test_list)
xyz = 52*63*53

def wrap_make_vec(args):
    return load_vec_and_tolil(*args)
# count_nonzero

def load_vec_and_tolil(num_ic, file_name):
    vec = sparse.load_npz(input_base_path + str(num_ic+1) + "/" + file_name).tolil()
    return vec

def multi_ok(job_args1, num_ic):
    with mp.get_context('spawn').Pool() as p:
        vecs = list(tqdm(p.imap(wrap_make_vec, job_args1), total=len(train_list)))

    lil_mtx = sparse.lil_matrix((train_num_records,xyz))
    for (row,vec) in enumerate(tqdm(vecs)):
        lil_mtx[row,:] = vec
        vecs[row] = []
    start = time.time()
    sparse.save_npz(out_base_path+"/train/IC/ic{}_matrix.npz".format(num_ic+1), lil_mtx.tocsr())
    elapsed_time = time.time() - start
    print ("save_elapsed_time:{0}".format(elapsed_time) + "[sec]")
    gc.collect()
    p.close()
    p.join()


if __name__== "__main__":
    num_ic = int(sys.argv[1])
    job_args1 = [(num_ic, file_name) for file_name in train_list]
    multi_ok(job_args1, num_ic)