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
    ic_matrix = lil_matrix((train_num_records, xyz))
    for (row,file_name_train) in enumerate(tqdm(train_list)):
        f = h5py.File(input_base_path+'/fMRI_train/'+file_name_train,'r')
        data = (f['SM_feature'])[...]
        vec = data[num_ic,:,:,:].reshape(xyz)
        lil_vec = lil_matrix(vec)
        ic_matrix[row,:] = lil_vec
    save_npz(out_base_path+"/train/IC/ic{}_matrix.npz".format(str(num_ic+1)), ic_matrix.tocsr())

        #f = h5py.File(input_base_path+'/fMRI_test/'+file_name_test,'r')
        #data = (f['SM_feature'])[...]
        #vec = data[num_ic,:,:,:].reshape(xyz)
        #lil_vec = lil_matrix(vec)
    
    #save_npz(out_base_path+"/test/vecs/{}/{}.npz".format(str(num_ic+1), file_name_test), coo_vec)
#p = mp.Pool(processes=multiprocessing.cpu_count() - 1)
def multi_ok(job_args1):
    with mp.Pool(processes=2) as p:
        list(p.imap(io_num_component, job_args1))
        p.close()
        p.terminate()
        p.join()
        gc.collect()
if __name__ == "__main__":
    job_args1 = [(num_ic) for num_ic in range(3, 54)]
    multi_ok(job_args1)

