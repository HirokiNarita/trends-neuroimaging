import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from cuml import SVR

from sklearn.preprocessing import StandardScaler

# tensorflowの警告抑制
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Ridge
class Model1Ridge:
    targets = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
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
        
        #self.model = xgb.train(params, dtrain, num_round, evals=watchlist)

    def predict(self, x):
        #data = xgb.DMatrix(x)
        pred = {}
        for target in targets:
            
        pred = self.age_model()
        return pred


# SVR
class Model1SVR:

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.scaler = StandardScaler()
        self.scaler.fit(tr_x)

        batch_size = 128
        epochs = 10

        tr_x = self.scaler.transform(tr_x)
        va_x = self.scaler.transform(va_x)
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(tr_x.shape[1],)))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam')

        history = model.fit(tr_x, tr_y,
                            batch_size=batch_size, epochs=epochs,
                            verbose=1, validation_data=(va_x, va_y))
        self.model = model

    def predict(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict_proba(x).reshape(-1)
        return pred


# 線形モデル
class Model2Ridge:

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.scaler = StandardScaler()
        self.scaler.fit(tr_x)
        tr_x = self.scaler.transform(tr_x)
        self.model = LogisticRegression(solver='lbfgs', C=1.0)
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict_proba(x)[:, 1]
        return pred
