# -*- coding: utf-8 -*-
"""
Created on Wed May  3 16:19:40 2023

@author: Wei zhang
@email 7201607004@stu.jiangnan.edu.cn
"""

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import average_precision_score, f1_score


def class_mAP(y_pred, y_true, labeled_n):
    # L1 = (y_true.data).cpu().numpy().reshape([-1])
    # L2 = (y_pred.data).cpu().numpy()
    # L1 = L1 - np.min(L1)+1
    # L2 = L2 - np.min(L2)+1

    if labeled_n > 2:
        APs = []
        for i in range(labeled_n):
            ap = average_precision_score(y_true[:, i], y_pred[:, i])
            APs.append(ap)
        return np.mean(APs)
    else:
        return average_precision_score(y_true, y_pred)


def class_mF1(y_pred, y_true, labeled_n):
    # L1 = (y_true.data).cpu().numpy().reshape([-1])
    # L2 = (y_pred.data).cpu().numpy()
    # L1 = L1 - np.min(L1)+1
    # L2 = L2 - np.min(L2)+1

    if labeled_n > 2:
        return f1_score(y_true, y_pred, average='macro')
    else:
        return f1_score(y_true, y_pred)


df = sio.loadmat('/Users/zhangwei/Documents/PythonWorkspace/RL_TSL/results/wine_rf.mat')

pred = df['best_pred']
pred_pro = df['best_pred_pro']
true = df['best_res']

mF1 = [0] * 5
mAP = [0] * 5

for i in range(5):
    fold_pred = pred[0][i]
    fold_pred_pro = pred_pro[0][i]
    fold_true = true[0][i]

    labeled_n = np.max(fold_true)
    mF1[i] = class_mF1(fold_pred, fold_true, labeled_n)

    if labeled_n > 2:
        end = OneHotEncoder()
        fold_true = end.fit_transform(fold_true).toarray()
        APs = []
        for i in range(labeled_n):
            ap = average_precision_score(fold_true[:, i], fold_pred_pro[:, i])
            APs.append(ap)
        mAP[i] = np.mean(APs)
    else:
        mAP[i] = average_precision_score(fold_true, fold_pred)

final_mF1 = np.mean(mF1)
final_mAP = np.mean(mAP)

final_mF1_std = np.std(mF1)
final_mAP_std = np.std(mAP)

print(final_mF1, final_mF1_std)
print(final_mAP, final_mAP_std)