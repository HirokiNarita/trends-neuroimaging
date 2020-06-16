import numpy as np
import h5py
import os
import gc

from tqdm import tqdm
import multiprocessing as mp
# scipy sparse
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from scipy.sparse import load_npz
from scipy.sparse import vstack
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix

# setting param
input_base_path = "/media/hiroki/working/kaggle_data/trends-neuroimaging/split_IC/test/vecs/"
list_path = "/media/hiroki/working/kaggle_data/trends-neuroimaging/split_IC/test/vecs/1"
out_base_path = "/media/hiroki/working/kaggle_data/trends-neuroimaging/split_IC"
#train_list = os.listdir(path=input_base_path+'/fMRI_train')
test_list = os.listdir(path=list_path)
#train_num_records = len(train_list)
test_num_records = len(test_list)
xyz = 52*63*53

def concat_num_component(num_ic):
    for (row,file_name) in enumerate(test_list):
        dok_mtx[row,:] = load_npz(input_base_path + str(num_ic+1) + "/" + file_name).todok()
        gc.collect()
    save_npz(out_base_path+"/test/IC/ic_{}.npz".format(str(num_ic+1)), dok_mtx.tocsr())

#p = mp.Pool(processes=multiprocessing.cpu_count() - 1)
def multi_ok(job_args1):
    with mp.get_context('spawn').Pool(4) as p:
        list(tqdm(p.imap(concat_num_component, job_args1), total=(53)))
        p.close()
        p.terminate()
        p.join()
        gc.collect()
     dok_mtx = dok_matrix((test_num_records,xyz))
if __name__ == "__main__":
    job_args1 = [(num_ic) for num_ic in range(3, 54)]
    multi_ok(job_args1)

