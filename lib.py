# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 13:49:38 2023

@author: Wei Zhang
@email: 7201607004@stu.jiangnan.edu.cn
"""

import numpy as np
import torch
import torch.nn.functional as func
from sklearn.metrics import precision_score, f1_score, accuracy_score, average_precision_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix, normalized_mutual_info_score, \
    adjusted_rand_score
from munkres import Munkres, print_matrix

def best_map(L1,L2):
    #L1 should be the labels and L2 should be the clustering number we got
    Label1 = np.unique(L1)       # 去除重复的元素，由小大大排列
    nClass1 = len(Label1)        # 标签的大小
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(int)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(int)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

def label2matrix(label):
    label = np.array(label)
    uq_la = np.unique(label)
    c = uq_la.shape[0]
    n = label.shape[0]
    label_mat = np.zeros((n, c))
    for i in range(c):
        index = (label == i)
        label_mat[index, i] = 1
    return label_mat

def pred_classes(X):
    _, pred_y = torch.max(func.softmax(X), 1)
    return pred_y


def class_mAP(y_pred, y_true, labeled_n):

    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()

    if labeled_n > 2:
        y_pred = label2matrix(y_pred)
        y_true = label2matrix(y_true)
        APs = []
        for i in range(labeled_n):
            ap = average_precision_score(y_true[:, i], y_pred[:, i], average='macro')
            APs.append(ap)
        return np.mean(APs)
    else:
        return average_precision_score(y_true, y_pred)


def class_mF1(y_pred, y_true, labeled_n):
    L1 = (y_true.data).cpu().numpy().reshape([-1])
    L2 = (y_pred.data).cpu().numpy()
    # L1 = L1 - np.min(L1)+1
    # L2 = L2 - np.min(L2)+1
    return f1_score(L2, L1, average='macro')
    # if labeled_n > 2:
    #     return f1_score(L2, L1, average='macro')
    # else:
    #     return f1_score(L2, L1)


def class_accuracy(y_pred, y_true, labeled_n):

    L1 = (y_true.data).cpu().numpy().reshape([-1])
    L2 = (y_pred.data).cpu().numpy()

    return accuracy_score(L2, L1)


# return (np.sum(L1 == L2)-labeled_n)/(L1.shape[0]-labeled_n)


def pred_cluster(X):
    _, pred_y = torch.max(X, 1)
    return pred_y


def cluster_accuracy(y_true, y_pred):
    # L1 = (y_true.data).cpu().numpy().reshape([-1])
    # L2 = (y_pred.data).cpu().numpy()
    # L1 = L1 - np.min(L1) + 1
    # L2 = L2 - np.min(L2) + 1
    # compute contingency matrix (also called confusion matrix)
    conti_matrix = contingency_matrix(y_true, y_pred)
    # find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-conti_matrix)
    # return cluster accuracy
    return conti_matrix[row_ind, col_ind].sum() / np.sum(conti_matrix)


def normalized_mutual_info(y_true, y_pred):
    return normalized_mutual_info_score(y_true, y_pred)


def purity_score(y_true, y_pred):
    # L1 = (y_true.data).cpu().numpy().reshape([-1])
    # L2 = (y_pred.data).cpu().numpy()
    # L1 = L1 - np.min(L1) + 1
    # L2 = L2 - np.min(L2) + 1
    # compute contingency matrix (also called confusion matrix)
    conti_matrix = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(conti_matrix, axis=0)) / np.sum(conti_matrix)


def adjusted_rand_index(y_true, y_pred):
    return adjusted_rand_score(y_true, y_pred)
