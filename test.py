# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 13:48:11 2019

@author: xq
"""
import numpy as np
from sklearn.metrics import roc_auc_score
a = np.array([0, 0, 0, 0, 0, 0, 0])
b = np.array([0, 0, 0, 0, 0, 0, 1])
print(roc_auc_score(b,a))

import numpy as np
from sklearn.metrics import roc_auc_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0, 0, 1, 1])
print(roc_auc_score(y_true, y_scores))