#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/29 17:44
# @Author  : liangxiao
# @Site    : 
# @File    : data_test.py
# @Software: PyCharm
import pandas as pd
if __name__ == '__main__':
    df_train = pd.read_csv("train1.csv")
    df_test = pd.read_csv("test.csv")
    print(len(df_train))
    df_1 = df_train[df_train['target'] == 1]
    df_0 = df_train[df_train['target']== 0]
    df = pd.concat([df_1,df_0[:30000]])
    df = df.sample(frac = 1).reset_index(drop=True)
    