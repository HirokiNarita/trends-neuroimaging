import numpy as np
import h5py
import os
import gc

from tqdm import tqdm
import multiprocessing as mp
# scipy sparse
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import save_npz
from scipy.sparse import coo_matrix

# setting param
input_base_path = "/media/hiroki/share/kaggle_data/trends-assessment-prediction"
out_base_path = "/media/hiroki/working/kaggle_data/trends-neuroimaging/split_IC"
train_list = os.listdir(path=input_base_path+'/fMRI_train')
test_list = os.listdir(path=input_base_path+'/fMRI_test')
train_num_records = len(train_list)
test_num_records = len(test_list)
xyz = 52*63*53

def io_num_component(num_ic):
    for file_name_train, file_name_test in zip(train_list,test_list):
        f = h5py.File(input_base_path+'/fMRI_train/'+file_name_train,'r')
        data = (f['SM_feature'])[...]
        vec = data[num_ic,:,:,:].reshape(xyz)
        coo_vec = coo_matrix(vec)
        save_npz(out_base_path+"/train/vecs/{}/{}.npz".format(str(num_ic+1), file_name_train), coo_vec)

        f = h5py.File(input_base_path+'/fMRI_test/'+file_name_test,'r')
        data = (f['SM_feature'])[...]
        vec = data[num_ic,:,:,:].reshape(xyz)
        coo_vec = coo_matrix(vec)
        save_npz(out_base_path+"/test/vecs/{}/{}.npz".format(str(num_ic+1), file_name_test), coo_vec)
#p = mp.Pool(processes=multiprocessing.cpu_count() - 1)
def multi_ok(job_args1):
    with mp.Pool() as p:
        list(tqdm(p.imap(io_num_component, job_args1), total=(53)))
        p.close()
        p.terminate()
        p.join()
        gc.collect()
if __name__ == "__main__":
    job_args1 = [(num_ic) for num_ic in range(0, 54)]
    multi_ok(job_args1)

