# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 17:12:44 2019

@author: drewqilong
"""

import pandas as pd
import numpy as np

from example import config

def load_data():
    df_train = pd.read_csv(config.TRAIN_FILE)
    df_train = df_train
    df_test = pd.read_csv(config.TEST_FILE)

    def preprocess(df):
        cols = [c for c in df.columns if c not in ["id", "target"]]
        # record every instance's missing feature number.
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
        # to get a new feature by multiply existing features.
        df["ps_car_13_x_ps_reg_03"] = df["ps_car_13"] * df["ps_reg_03"]
        return df

#    df_train = preprocess(df_train)
#    df_test = preprocess(df_test)
    
    
    ''' One hot for categorical data'''
    df_train = pd.get_dummies(df_train, columns=config.CATEGORICAL_COLS)
    df_test = pd.get_dummies(df_test, columns=config.CATEGORICAL_COLS)
    
    cols = [c for c in df_train.columns if c not in ["id", "target"]]
    cols = [c for c in cols if c not in config.IGNORE_COLS]
    
    X_train = df_train[cols].values
#    X_test = df_test[cols].values
    X_test = None
    y_train = df_train["target"].values
#    ids_test = df_test["id"].values
    ids_test = None
    cat_features_indices = [i for i, c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return df_train, df_test, X_train, y_train, X_test, ids_test, cat_features_indices

def load_ctr_data():
    df_train = pd.read_csv(config.TRAIN_CTR_FILE)
    df_test = None
    ''' One hot for categorical data'''
    df_train = pd.get_dummies(df_train)
    ''' Flag all the NaN to 0'''
    df_train[df_train.select_dtypes(include=['float64','float32']).columns] = \
    df_train[df_train.select_dtypes(include=['float64','float32']).columns].fillna(value = '0.0')
    cols = [c for c in df_train.columns if c not in ["Id", "Label"]]

    X_train = df_train[cols].values
    X_test = None
    y_train = df_train["Label"].values
    ids_test = None
    cat_features_indices = [i for i, c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return df_train, df_test, X_train, y_train, X_test, ids_test, cat_features_indices