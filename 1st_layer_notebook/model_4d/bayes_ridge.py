import os
import gc
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
from contextlib import contextmanager
import time

import numpy as np
import pandas as pd
import scipy as sp
import random

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

import bayes_ridge_functions as func
#===========================================================
# Config
#===========================================================
OUTPUT_DICT = ''
ID = 'Id'
TARGET_COLS = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
SEED = 2020
N_FOLD = 5

for num in range(1,54):
    print("IC{}".format(num))
    # Data load
    base_path = '/media/hiroki/share/kaggle_data/trends-assessment-prediction'
    out_base_path = '/media/hiroki/working/kaggle_data/trends-neuroimaging/pred_by_ic'+'/ic_{}'.format(num)
    os.makedirs(out_base_path+'/ic_{}'.format(num))
    train = pd.read_csv(base_path+'/train_scores.csv', dtype={'Id':str})
    sample_submission = pd.read_csv(base_path+'/sample_submission.csv', dtype={'Id':str})
    # make_test
    sample_submission['ID_num'] = sample_submission[ID].apply(lambda x: int(x.split('_')[0]))
    test = pd.DataFrame({ID: sample_submission['ID_num'].unique().astype(str)})
    del sample_submission['ID_num']; gc.collect()
    # load_ic
    ic_path = "/media/hiroki/working/kaggle_data/trends-neuroimaging/split_IC/svd"
    ic = np.load(ic_path+'/ic_{}.npz.npy'.format(num))
    ic_train = ic[:5877, :]
    ic_test = ic[5877:, :]
    del ic
    ic_train = pd.concat([train["Id"], pd.DataFrame(ic_train)], axis=1)
    ic_test = pd.concat([test["Id"], pd.DataFrame(ic_test)], axis=1)
    # merge
    train = train.merge(ic_train, on=ID, how='left')
    test = test.merge(ic_test, on=ID, how='left')

    pred_train_targets_bayes_ridge = {}
    pred_test_targets_bayes_ridge = {}

    pred_train_targets_bayes_ridge_dfs = {}
    pred_test_targets_bayes_ridge_dfs = {}

    overal_score = 0

    for target, w in [("age", 0.3),
                      ("domain1_var1", 0.175),
                      ("domain1_var2", 0.175),
                      ("domain2_var1", 0.175),
                      ("domain2_var2", 0.175)]:
    
        train_df = train[train[target].notnull()]
        test_df = test
    
        use_idx = train_df.index
        train_x = train_df.drop([ID]+TARGET_COLS, axis=1)
        train_y = train_df[target]
        test_x = test_df.drop(ID, axis=1)
        bayes_ridge = BayesianRidge(n_iter = 3000)
    
        print("-----{}-----".format(target))
        pred_train, preds_test, score_cv = func.predict_cv(train_x, train_y, test_x, bayes_ridge, target)
        overal_score += w*score_cv
        pred_train_targets_bayes_ridge[target] = pred_train
        pred_test_targets_bayes_ridge[target] = preds_test
    
        pred_train_targets_bayes_ridge_dfs[target] = pd.Series(pred_train, name="pre_bayRidge_4D_IC{}_{}".format(num, target), index=use_idx)
        pred_train_targets_bayes_ridge_dfs[target] = pd.merge(train['Id'],
                                                     pred_train_targets_bayes_ridge_dfs[target],
                                                     left_index=True,
                                                     right_index=True)
        pred_test_targets_bayes_ridge_dfs[target] = pd.Series(preds_test, name="pre_bayRidge_4D_IC{}_{}".format(num, target))
        pred_test_targets_bayes_ridge_dfs[target] = pd.concat([test['Id'], pred_test_targets_bayes_ridge_dfs[target]], axis=1)
    print('--------------------------------------------')
    print("Overal score:", np.round(overal_score, 8))

    for i, pred_df in enumerate(pred_train_targets_bayes_ridge_dfs.values()):
        if i == 0:
            pred_1st_train_df = pred_df
        else:
            pred_1st_train_df = pd.concat([pred_1st_train_df, pred_df.drop("Id", axis=1)], axis=1)

    for i, pred_df in enumerate(pred_test_targets_bayes_ridge_dfs.values()):
        if i == 0:
            pred_1st_test_df = pred_df
        else:
            pred_1st_test_df = pd.concat([pred_1st_test_df, pred_df.drop("Id", axis=1)], axis=1)
    
    pred_1st_train_df.to_csv(out_base_path+"/pred_train.csv")
    pred_1st_test_df.to_csv(out_base_path+"/pred_test.csv")