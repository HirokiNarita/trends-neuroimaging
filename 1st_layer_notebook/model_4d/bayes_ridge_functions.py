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
#===========================================================
# Config
#===========================================================
OUTPUT_DICT = ''
ID = 'Id'
TARGET_COLS = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
SEED = 2020
N_FOLD = 5

def metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))

# 学習データに対する「目的変数を知らない」予測値と、テストデータに対する予測値を返す関数
def predict_cv(train_x, train_y, test_x, model, target_name):
    preds = []
    preds_test = []
    va_idxes = []
    
    score = []
    #mae = []
    #rmse = []
    # shuffleしなくても良い
    kf = KFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
    ss = StandardScaler()
    # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する
    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
        tr_x, va_x = train_x.iloc[tr_idx].values, train_x.iloc[va_idx].values
        tr_y, va_y = train_y.iloc[tr_idx].values, train_y.iloc[va_idx].values
        # z-scaling X
        tr_x = ss.fit_transform(tr_x)
        va_x = ss.transform(va_x)
        test_x = ss.transform(test_x)
        model.fit(tr_x, tr_y)
        pred = model.predict(va_x)
        preds.append(pred)
        pred_test = model.predict(test_x)
        preds_test.append(pred_test)
        va_idxes.append(va_idx)
        
        score.append(metric(va_y, pred))
        #mae.append(mean_absolute_error(va_y, pred))
        #rmse.append(np.sqrt(mean_squared_error(va_y, pred)))
        
    score_cv = np.array(score).mean()
    #mae_cv = np.array(mae).mean()
    #rmse_cv = np.array(rmse).mean()
    print("{0}_score:{1}".format(target_name, np.round(score_cv, 8)))
    #print("{0}_mae:{1}".format(target_name, np.array(mae_cv).mean()))
    #print("{0}_rmse:{1}".format(target_name, np.array(rmse_cv).mean()))
    # バリデーションデータに対する予測値を連結し、その後元の順序に並べ直す
    va_idxes = np.concatenate(va_idxes)
    preds = np.concatenate(preds, axis=0)
    order = np.argsort(va_idxes)
    pred_train = preds[order]

    # テストデータに対する予測値の平均をとる
    preds_test = np.mean(preds_test, axis=0)

    return pred_train, preds_test, score_cv