# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 10:53:20 2019

@author: drewqilong
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/29 17:19
# @Author  : liangxiao
# @Site    : 
# @File    : main.py
# @Software: PyCharm
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from example.data_reader import DataParser
from example.data_reader import FeatureDictionary
from example import config
from example.log import log
from load_data import *





if __name__ == '__main__':
    log("start to load data...")
    df_train, df_test, X_train, y_train, X_test, ids_test, cat_features_indices = load_ctr_data()

    # folds
    log("split folds")
    folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                                 random_state=config.RANDOM_SEED).split(X_train, y_train))



    '''#3.Random Forest Classifier'''
    seed = 43
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(random_state = seed, n_estimators = 100)
    '''#6.Decision Tree Classifier'''
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(random_state = seed)    
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    '''#9.ExtraTrees Classifier'''
    from sklearn.ensemble import ExtraTreesClassifier
    etc = ExtraTreesClassifier(random_state = seed)

    _get = lambda x, l: [x[i] for i in l]

    def train_accuracy(model):
        model.fit(X_train_, y_train_)
        train_accuracy = model.score(X_train_, y_train_)
        train_accuracy = np.round(train_accuracy*100, 2)
        valid_accuracy = model.score(X_valid_, y_valid_)
        valid_accuracy = np.round(valid_accuracy*100, 2)
        print(" train-result=%.4f, valid-result=%.4f" % ( train_accuracy, valid_accuracy))
       
    for i,(train_idx, valid_idx) in enumerate(folds):
        # get train/valid sets vial row sampling based on k-folds validation
        log("in fold [ %s ]..."%i)
        
        X_train_, y_train_ = _get(X_train, train_idx), _get(y_train, train_idx)
        X_valid_, y_valid_ = _get(X_train, valid_idx), _get(y_train, valid_idx)
#    n = 1500
#    X_train_ = X_train[:n]
#    y_train_ = y_train[:n]
#    X_valid_ = X_train[n:]
#    y_valid_ = y_train[n:]
#    X_train_ = np.array(X_train_)
#    X_valid_ = np.array(X_valid_)
        train_accuracy(rf)
           

        
