import numpy as np
import h5py
import os
import gc

from tqdm import tqdm
import multiprocessing as mp
# scipy sparse
from scipy.sparse import coo_matrix
from scipy.sparse import save_npz

# setting param
input_base_path = "/media/hiroki/share/kaggle_data/trends-assessment-prediction"
out_base_path = "/media/hiroki/working/kaggle_data/trends-neuroimaging/split_IC"
#train_list = os.listdir(path=input_base_path+'/fMRI_train')
test_list = os.listdir(path=input_base_path+'/fMRI_test')
#train_num_records = len(train_list)
test_num_records = len(test_list)
xyz = 52*63*53

def io_num_component(num_ic):
    for file_name in train_list:
        f = h5py.File(input_base_path+'/fMRI_train/'+file_name,'r')
        data = f['SM_feature']
        coo_vec = coo_matrix(data[num_ic,:,:,:].reshape(xyz))
        save_npz(out_base_path+"/train/vecs/{}/ic_{}.npz".format(str(num_ic+1), file_name), coo_vec)

#p = mp.Pool(processes=multiprocessing.cpu_count() - 1)
def multi_ok(job_args1):
    with mp.get_context('spawn').Pool() as p:
        list(tqdm(p.imap(io_num_component, job_args1), total=(53)))
        p.close()
        p.terminate()
        p.join()
        gc.collect()
if __name__ == "__main__":
    job_args1 = [(num_ic) for num_ic in range(12, 54)]
    multi_ok(job_args1)

