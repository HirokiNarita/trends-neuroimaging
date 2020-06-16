import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from cuml import SVR

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.metrics import make_scorer

def MAPE(y_true, y_pred, **kwargs):
    '''Returns MAPE between y_true and y_pred'''
    return np.sum(np.abs(y_true - y_pred)) / y_true.sum()

mape_scorer = make_scorer(MAPE, greater_is_better=False)

target_columns = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']

# Ridge
class Model1Ridge:
    def __init__(self):
        self.age_model = None
        self.d1v1_model = None
        self.d1v2_model = None
        self.d2v1_model = None
        self.d2v2_model = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        model = Ridge()
        cv = KFold(n_splits = 15, shuffle=True, random_state=2020)
        grid = {
            'alpha': [0.0003, 0.001, 0.003, 0.01, 0.03]
        }
        gs = GridSearchCV(model, grid, n_jobs=-1, cv=cv, verbose=0, scoring=mape_scorer)
        
        best_models = {}
        for col in target_columns:
            #X_train = data.dropna(subset=[col], axis=0).drop(list(target_columns), axis=1).drop('Id', axis=1)
            X_train[fnc_columns] /= 500
            y_train = data.dropna(subset=[col], axis=0)[col]
            gs.fit(tr_x, tr_y)
            best_models[col] = gs.best_estimator_
        #self.model = xgb.train(params, dtrain, num_round, evals=watchlist)

    def predict(self, x):
        for col in target_columns:
            test_prediction[col] = best_models[col].predict(x)
        return pred