# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 12:18:30 2023

@author: Wei Zhang
@email: 7201607004@stu.jiangnan.edu.cn
"""

import numpy as np
from fuzzy_clustering import ESSC, FuzzyCMeans
import math
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
from scipy.special import softmax
from numpy import linalg as la
import time


device = 'cpu'#torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class Diff_FNN(nn.Module):
    def __init__(self, data, L, G, Y, n_rules, n_layers, cluster_m=2, cluster_eta=0.1,
                 cluster_gamma=0.1, cluster_scale=1):
        
        """
        :param n_rules: number of rules
        :param cluster_m: fuzzy index $m$ for ESSC
        :param cluster_eta: parameter $\eta$ for ESSC
        :param cluster_gamma: parameter $\gamma$ for ESSC
        :param cluster_scale: scale parameter for ESSC
        :param L: similar matrix
        """
        
        super(Diff_FNN, self).__init__()
        
        self.n_cluster = n_rules
        self.cluster_m = cluster_m
        self.cluster_eta = cluster_eta
        self.cluster_gamma = cluster_gamma
        self.cluster_scale = cluster_scale
        self.layers = n_layers
        self.L = L
        self.G = G
        self.Y = Y
        self.center = 0
        self.var = 0
        self.xp = torch.tensor(self.fuzzy_layer_train(data)).type('torch.FloatTensor')
        self.xp = self.xp.to(device=device)
        self.dim = self.xp.shape[1]
        self.lam = nn.Parameter(torch.FloatTensor([0]), requires_grad=True).to(device)
        self.classes = self.Y.shape[1]
        self.fc1 = nn.ModuleList()

        for k in range(self.layers):
            self.fc1.append(nn.Linear(self.dim, self.dim, bias=False, device=device))

    
    def fuzzy_layer_train(self, X):
        
        self.center, self.var = self.__cluster__(X, self.n_cluster, self.cluster_eta, self.cluster_gamma, self.cluster_scale)
        mem = self.__firing_level__(X, self.center, self.var)
        Xp = self.x2xp(X, mem)
        
        return Xp
    
    def fuzzy_layer_test(self, X):
        mem = self.__firing_level__(X, self.center, self.var)
        return self.x2xp(X, mem)

    def active(self, x, thershold):
        tempa = x - thershold
        tempb= -1.0 * x - thershold
        loss = func.relu(tempa).to(device) - func.relu(tempb).to(device)
        return loss


    def init_para(self):

        P_new = list()
        IW_list = list()
        lamIn = torch.eye(self.dim, device=device)
        torch.manual_seed(2023)
        P0 = torch.rand(self.classes, self.dim, device=device)
        alpha = 0.0001
        for k in range(self.layers):
            
            temp_A = torch.bmm(torch.bmm(self.xp.T.unsqueeze(0), self.L.unsqueeze(0)), self.xp.unsqueeze(0))
            temp_A = temp_A.sum(0)
            temp_A = func.normalize(temp_A, dim=1)
            _,sigma,_ = la.svd(temp_A.cpu())

            temp_B = torch.bmm(torch.bmm(torch.bmm(self.xp.T.unsqueeze(0), self.G.T.unsqueeze(0)), self.G.unsqueeze(0)), self.xp.unsqueeze(0))
            temp_B = temp_B.sum(0)
            temp_B = func.normalize(temp_B, dim=1)
            
            beta = torch.tensor( max(sigma) ) + torch.norm(temp_B, p='fro')
            init_weight = lamIn-(1/beta)*temp_A -(1/beta)*temp_B

            # --ablation part--
            # beta = torch.tensor(max(sigma))
            # init_weight = lamIn - (1 / beta) * temp_A
            
            self.fc1[k].weight = torch.nn.Parameter(init_weight)
            IW_list.append(init_weight)
            Ln = nn.LayerNorm([self.classes, self.dim], elementwise_affine=True, device=device)
            if k == 0:
                temp = self.active(Ln(self.fc1[k](P0)), thershold=alpha/beta)
            else:
                P = self.active(Ln(self.fc1[k](temp)), thershold=alpha/beta)
                temp = P
            
            P_new.append(torch.tensor(temp.cpu().detach().numpy()).type('torch.FloatTensor'))
        
        return P_new[-1], alpha/beta, self.center, self.var
    
    
    def forward(self, P0):

        for k in range(self.layers):
            
            if k == 0:
                # P.append(self.active(self.fc1[k](P0.T).T, self.lam))
                temp_P = self.active(self.fc1[k](P0), self.lam)
            else:
                P = self.active(self.fc1[k](temp_P), self.lam)
                temp_P = P

            # func.normalize

        return self.xp, P, self.lam
       
    @staticmethod
    def __cluster__(data, n_cluster, eta, gamma, scale):
        """
        Comute data centers and membership of each point by ESSC, and compute the variance of each feature
        :param data: n_Samples * n_Features
        :param n_cluster: number of center
        :return: centers: data center, delta: variance of each feature
        """
        fuzzy_cluster = ESSC(n_cluster, eta=eta, gamma=gamma, tol_iter=100, scale=scale).fit(data)
        # fuzzy_cluster1 = FuzzyCMeans(n_cluster)
        centers = fuzzy_cluster.center_
        delta = fuzzy_cluster.variance_
        return centers, delta

    @staticmethod
    def __firing_level__(data, centers, delta):
        """
        Compute firing strength using Gaussian model
        :param data: n_Samples * n_Features
        :param centers: data center，n_Clusters * n_Features
        :param delta: variance of each feature， n_Clusters * n_Features
        :return: firing strength
        """
        d = -(np.expand_dims(data, axis=2) - np.expand_dims(centers.T, axis=0))**2 / (2 * delta.T)
        d = np.exp(np.sum(d, axis=1))
        d = np.fmax(d, np.finfo(np.float64).eps)
        return d / np.sum(d, axis=1, keepdims=True)

    @staticmethod
    def x2xp(X, mem, order=1):
        """
        Converting raw input feature X to TSK consequent input
        :param X: raw input, [n_sample, n_features]
        :param mem: firing level of each rule, [n_sample, n_clusters]
        :param order:order of TSK, 0 or 1
        :return:
        """
        if order == 0:
            return mem
        else:
            N = X.shape[0]
            mem = np.expand_dims(mem, axis=1)
            X = np.expand_dims(np.concatenate((X, np.ones([N, 1])), axis=1), axis=2)
            X = np.repeat(X, repeats=mem.shape[1], axis=2)
            xp = X * mem
            xp = xp.reshape([N, -1])
            return xp


def calssification_loss(Xp, P, Y):
    # lsoft = nn.LogSoftmax(dim=1)
    res = Xp.mm(P.T).to(device)
    Y = Y.to(device)
    # criterion = nn.CrossEntropyLoss()
    # loss1 = criterion(res, Y)
    loss1 = torch.norm(func.softmax(res) - Y, p=2)
    # loss2 = torch.norm(res.mm(res.T) - X_tr.mm(X_tr.T), p=2) 0.001

    total_loss = loss1 + 1e-3*torch.norm(P, p=2)
    return total_loss



