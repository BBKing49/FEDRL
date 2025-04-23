# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 20:07:33 2023

@author: Wei Zhang
@email: 7201607004@stu.jiangnan.edu.cn
"""

import os
import sys
import time
import torch
import logging
import argparse
import scipy.io as sio
import numpy as np
import random
import torch.optim as optim
from Diff_FNN_for_classification import Diff_FNN, calssification_loss
from lib import pred_classes, class_accuracy, class_mAP, class_mF1
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings('ignore')

# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#
# setup_seed(20)

device = 'cpu'  # torch.device("mps" if torch.backends.mps.is_available() else "cpu")

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)

rule_nums = [6] # np.arange(2, 12, 2)
epoch_nums = [1500]  # np.arange(100,1500,500)
layer_nums = [20]
learning_rates = [1e-5]  # , 0.001, 0.01, 0.1
datasets = ['IS']  # 'IS','PAG','aution_cals','magic04','NATICUS_cla','USPS_gist_zw',

T = time.time()
for dataset in datasets:
    best_acc = 0
    flag = 0

    save_res_path = 'res_ablation/{}.mat'.format(dataset)

    df = sio.loadmat('Data/' + dataset + '.mat')
    X = df['data']
    Y = df['labels']
    lab = np.argmax(Y, 1)

    minmax = MinMaxScaler()
    X = minmax.fit_transform(X)
    skflods = StratifiedKFold(n_splits=5, random_state=2023, shuffle=True)

    for rule in rule_nums:
        for epoch_num in epoch_nums:
            for layer_num in layer_nums:
                for lr in learning_rates:

                    print("rule:{:d}, epoch:{:d}, layder:{:d}, lr:{:f}".format(rule, epoch_num, layer_num, lr))

                    parser = argparse.ArgumentParser("Tsk Fuzy NN")
                    parser.add_argument('--dataset', type=str, default=dataset, help='dataset name')
                    parser.add_argument('--learning_rate', type=float, default=lr, help='init learning rate')
                    # parser.add_argument('--lr_decay', type=float, default=0.99, help='learning_rate decay')
                    # parser.add_argument('--lr_decay_gap', type=int, default=2, help='learning rate decay epoch')
                    parser.add_argument('--report_freq', type=float, default=5, help='report frequency')
                    parser.add_argument('--method', type=int, default=2, help='loss calculation method')
                    parser.add_argument('--epochs', type=int, default=epoch_num, help='num of training epochs')
                    parser.add_argument('--layers', type=int, default=layer_num, help='total number of layers')
                    parser.add_argument('--log_root', type=str, default='log/classification', help='log root directory')
                    args = parser.parse_args()

                    if not os.path.isdir(os.path.join(args.log_root, args.dataset)):
                        os.makedirs(os.path.join(args.log_root, args.dataset))
                    if flag == 0:
                        fh = logging.FileHandler(
                            os.path.join(args.log_root, args.dataset, '_{}.txt'.format(time.strftime("%Y%m%d_%H%M%S"))))
                        fh.setFormatter(logging.Formatter(log_format))
                        logger = logging.getLogger()
                        logger.addHandler(fh)
                        flag = 1

                    acc = [0] * 5
                    mF1 = [0] * 5
                    mAP = [0] * 5

                    time_lg = [0] * 5
                    time_train = [0] * 5

                    for idx, (train_idx, test_idx) in enumerate(skflods.split(X, lab)):

                        T_LG_s = time.time()

                        X_tr = torch.tensor(X[train_idx]).type('torch.FloatTensor')
                        Y_tr = torch.tensor(Y[train_idx]).type('torch.FloatTensor')
                        X_te = torch.tensor(X[test_idx]).type('torch.FloatTensor')
                        Y_te = torch.tensor(Y[test_idx]).type('torch.FloatTensor')

                        sim_mat = cos_sim(X_tr)
                        L = np.diag(sum(sim_mat, 2)) - sim_mat
                        G = np.ones(X_tr.shape[0]) - sim_mat

                        # sim_mat = torch.cosine_similarity(X_tr, X_tr)
                        # L = torch.diag_embed(torch.sum(sim_mat,2)) - sim_mat
                        # G = torch.ones(X_tr.shape[0]) - sim_mat

                        L = torch.tensor(L).type('torch.FloatTensor')
                        L = L.to(device)
                        G = torch.tensor(G).type('torch.FloatTensor')
                        G = G.to(device)

                        T_LG_f = time.time()

                        time_lg[idx] = (T_LG_f - T_LG_s)
                        model = Diff_FNN(X_tr, L, G, Y_tr, n_rules=rule, n_layers=args.layers)

                        model.to(device)
                        P0, orgin_lam, center, var = model.init_para()
                        P0 = P0.to(device)
                        n_classes = Y_tr.shape[1]
                        loss_cal = list()

                        T_model_s = time.time()
                        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
                        for epoch in range(args.epochs):
                            # if args.learning_rate > 1e-9:
                            #     optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
                            # else:
                            #     args.learning_rate = 0.1
                            #     optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

                            optimizer.zero_grad()
                            # Forward
                            [Xp_tr, P, learn_lam] = model(P0)
                            loss_list = list()
                            total_loss = 0
                            final_loss = calssification_loss(Xp_tr, P, Y_tr)
                            # for k in range(args.layers):
                            #     loss_list.append( calssification_loss(Xp_tr, P[k], Y_tr) )
                            #     total_loss = total_loss + loss_list[-1]
                            # if args.method == 1:
                            #     final_loss = total_loss
                            # else:
                            #     final_loss = loss_list[-1] # now

                            loss_cal.append(float(final_loss.cpu().detach().numpy()))
                            t_loss_f = time.time()
                            final_loss.backward(retain_graph=True)
                            optimizer.step()

                            # if epoch>1:
                            #     if np.abs(loss_cal[-1]-loss_cal[-2])<1e-6:
                            #         break
                        # plt.plot(loss_cal, linewidth=3.0)
                        # plt.xticks(fontsize=10)
                        # plt.yticks(fontsize=10)
                        # plt.xlabel('Iteration', fontsize=14)
                        # plt.ylabel('Objection function values', fontsize=14)
                        # plt.title('NATICU',fontsize=16)
                        # plt.show()
                        T_model_f = time.time()
                        time_train = (T_model_f - T_model_s)
                        Xp_te = model.fuzzy_layer_test(X_te)
                        Xp_te = torch.tensor(Xp_te, dtype=torch.float32).to(device)
                        pred_y = pred_classes(Xp_te.mm(P.T))
                        acc[idx] = class_accuracy(pred_y, pred_classes(Y_te), n_classes)
                        # mAP[idx] = class_mAP(pred_y, pred_classes(Y_te), n_classes)
                        mF1[idx] = class_mF1(pred_y, pred_classes(Y_te), n_classes)

                    print("Lg time:{:f}, trian time:{:f}".format(np.mean(time_lg), np.mean(time_train)))

                    if np.mean(acc) > best_acc:
                        best_acc = np.mean(acc)
                        fin_acc = acc
                        fin_mF1 = mF1
                        fin_mAP = mAP

                        # torch.save(model,'./model/'+args.dataset+'_mdoel')

                        print("-------------\n")
                        logging.info('dataset: {:s}'.format(args.dataset))
                        logging.info('epoch %d lr %e', epoch, lr)
                        logging.info('rule_nums: {:e}'.format(rule))
                        logging.info('layer_nums: {:e}'.format(layer_num))
                        logging.info('loss: {:e}'.format(final_loss))
                        logging.info('acc: {:e}'.format(best_acc))
                        logging.info('acc_std: {:e}'.format(np.std(acc)))
                        # logging.info('mAP: {:e}'.format( np.mean(mAP)))
                        # logging.info('mAP_std: {:e}'.format(np.std(mAP)))
                        logging.info('mF1: {:e}'.format(np.mean(mF1)))
                        logging.info('mF1_std: {:e}'.format(np.std(mF1)))
                        print("\n")

                        # sio.savemat(save_res_path, {'best_acc': fin_acc, 'best_mF1': fin_mF1})
                    del parser





