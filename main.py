import argparse
from pathlib import Path
from pprint import pprint
import os
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, accuracy_score

import optuna

from preprocess import preprocess
from configs import *

def objective(trial, X_train, y_train):
    
    kernel = trial.suggest_categorical('kernel', ['linear','rbf','poly'])
    gamma = trial.suggest_loguniform('gamma',1e-5,1e5)
    C = trial.suggest_loguniform('C',1e-5,1e5)

    svc = SVC(kernel=kernel, gamma=gamma, C=C)

    score = cross_val_score(svc, X_train, y_train, cv=2, scoring="accuracy")

    acc_mean = score.mean()

    return acc_mean

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='Baseline',
                        help="config name in configs.py")
    opt = parser.parse_args()
    pprint(opt)

    ''' Configure path '''
    cfg = eval(opt.config)
    export_dir = Path('output') / cfg.name
    export_dir.mkdir(parents=True, exist_ok=True)

    ''' Prepare data '''
    train = pd.read_csv(cfg.train)
    test = pd.read_csv(cfg.test)
    train = train.replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)
    test = test.replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)
    
    train = preprocess(train, train)
    test = preprocess(test, train)

    train_data = train.values
    X = train_data[:, 2:]
    y  = train_data[:, 1]
    test_data = test.values
    X_test = test_data[:, 1:]

    RS = RobustScaler()
    RS.fit(X)
    X = RS.transform(X)
    X_test = RS.transform(X_test)
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    acc_trn_list = []
    acc_tst_list = []

    k=0

    ''' Opuna phase'''
    
    for train_itr, test_itr in kfold.split(X, y):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X[train_itr], y[train_itr]), n_trials=1)
        print(f'Fold #{k+1}; Best Parameter: {study.best_params}, Accuracy: {study.best_value}')
        
        ''' Train on best params '''
        best_svc = SVC(**study.best_params)

        best_svc.fit(X[train_itr], y[train_itr])

        ''' Inference '''
        trn_score = best_svc.predict(X[train_itr])
        tst_score = best_svc.predict(X[test_itr])

        ''' Calc ACC scores '''
        acc_trn_list.append(accuracy_score(y[train_itr], trn_score))
        acc_tst_list.append(accuracy_score(y[test_itr], tst_score))
        k=k+1
