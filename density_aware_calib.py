# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import optimize
from sklearn.isotonic import IsotonicRegression

### for calibration net
import torch
import torch.nn as nn
from tqdm import tqdm

import time
import gc

# pip install faiss-cpu --no-cache
# conda install faiss-gpu cudatoolkit=11.0 -c pytorch
# https://github.com/facebookresearch/faiss/issues/821
import faiss
from sklearn.cluster import KMeans

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def to_onehot(arr, n_class):
    return np.eye(n_class)[arr]

def label2onehot(label):
    if len(label.shape) == 1:
        n_class = int(label.max() + 1)
        label = to_onehot(label, n_class)
    if len(label.shape) == 2 and label.shape[1] == 1:
        n_class = int(label.max() + 1)
        label = to_onehot(label[:,0], n_class)
    return label

def np_softmax(x):
    max = np.max(
        x, axis=1, keepdims=True
    )  # returns max of each row and keeps same dims
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(
        e_x, axis=1, keepdims=True
    )  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x


class KNNScorer(object):
    def __init__(self, 
        # method, 
        top_k=1, 
        avg_top_k=False,  
        return_dist_arr=False 
    ):
        """
        top_k: 
            Pick top-k distance value as a measurement of density
        avg_top_k: 
            if average top-k distances. 
            Default: pick k-th distance value.
        return_dist_arr:
            if return distance matrix, instead of one value for each sample.
        """
        self.avg_top_k = avg_top_k
        self.top_k = top_k
        self.return_dist_arr = return_dist_arr

        # knn
        self.ftrain_list = []

    def get_score(self, test_logits, test_feats):
        return self.knn_score(test_feats)

    def set_train_feat(self, train_feats, train_labels, class_num, _type="single"):
        normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
        prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))# Last Layer only

        if _type == "single":
            train_feats = [train_feats]
        
        for train_feat in train_feats:
            ftrain = prepos_feat(train_feat.astype(np.float32))
            self.ftrain_list += [ftrain]
            print(f"Set train features to KNNScorer: shape {train_feat.shape}")

        del train_feats
        gc.collect()


    def knn_score(self, test_feats):
        """
        test_feats: List of features (N, vector_length).
            Test features extracted from the classifier.
        """
        normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
        prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))# Last Layer only
            
        assert len(test_feats) == len(self.ftrain_list)
        ftrain_list = self.ftrain_list
        ftest_list = [
            prepos_feat(feat.astype(np.float32)) 
            for feat in test_feats
        ]

        D_list = []
        ood_scores = np.zeros((len(test_feats), test_feats[0].shape[0]))
        for i, (ftrain, ftest) in enumerate(zip(ftrain_list, ftest_list)):
            index = faiss.IndexFlatL2(ftrain.shape[1])
            index.add(ftrain)
            D, _ = index.search(ftest, self.top_k)
            D_list.append(D[:, -1000:])
            if self.avg_top_k:
                ood_scores[i, :] = D[:,-self.top_k:].mean(1)
            else:
                ood_scores[i, :] = D[:,-self.top_k]
        
        if self.return_dist_arr:
            return D_list
    
        print("ood scores shape: ", ood_scores.shape)
        return ood_scores




class DAC(object):
    def __init__(self, 
        ood_values_num=1,   
        tol=1e-12, 
        eps=1e-7, 
        disp=False, # to print optimization process
        ):
        """
        T = (w_i * knn_score_i) + w0 
        p = softmax(logits / T)
        """
        self.method = "L-BFGS-B"

        self.ood_values_num = ood_values_num
        print("ood_values_num: ", self.ood_values_num)

        self.tol = tol 
        self.eps = eps 
        self.disp = disp

        self.bnds = [[0, 10000.0]] * self.ood_values_num + [[-100.0, 100.0]]
        self.init = [1.0] * self.ood_values_num + [1.0] 
       

    def get_temperature(self, w, outlier_p):
        if self.ood_values_num == 1:
            if type(outlier_p).__module__ == np.__name__:
                if len(outlier_p.shape) == 1:
                    outlier_p = [outlier_p]
                else:
                    outlier_p = [outlier_p[i, :] for i in range(outlier_p.shape[0])]
                
        assert len(outlier_p) == self.ood_values_num, (outlier_p, len(outlier_p), self.ood_values_num)
        
        if len(outlier_p) != 0:
            sample_size = len(outlier_p[0])
            t = np.zeros(sample_size)
            
            for i in range(self.ood_values_num):
                t += w[i] * outlier_p[i] 
            t += w[-1]
        else:
            # temperature scaling
            t = np.zeros(1)
            t += w[-1]

        # return t
        # temperature should be a positive value
        return np.clip(t, 1e-20, None)

    def mse_lf(self, w, *args):
        ## find optimal temperature with MSE loss function
        logit, label, outlier_p = args
        t = self.get_temperature(w, outlier_p)
        logit = logit / t[:, None]
        p = np_softmax(logit)
        mse = np.mean((p - label) ** 2)
        return mse

    def ll_lf(self, w, *args):
        ## find optimal temperature with Cross-Entropy loss function
        logit, label, outlier_p = args
        t = self.get_temperature(w, outlier_p)
        logit = logit / t[:, None]
        p = np_softmax(logit)
        N = p.shape[0]
        ce = -np.sum(label * np.log(p + 1e-12)) / N
        return ce

    def optimize(self, logit, label, outlier_p, loss="ce"):
        """
        logit (N, C): classifier's outputs before softmax
        label (N, C): true labels, one-hot
        outlier_p (N, number_of_layers): 
            the value that represents how far the sample is in the feature space.
            we use KNN scoring strategy.
        """

        if not isinstance(self.eps, list):
            self.eps = [self.eps]

        if loss == "ce":
            func = self.ll_lf
        elif loss == "mse":
            func = self.mse_lf
        else:
            raise NotImplementedError

        
        # func:ll_t, 1.0:initial guess, args: args of the func, ..., tol: tolerence of minimization
        params = optimize.minimize(
            func,
            self.init,
            args=(logit, label, outlier_p),
            method=self.method,
            bounds=self.bnds,
            tol=self.tol,
            options={"eps": self.eps, "disp": self.disp},
        )
        # print(params)
        w = params.x
        print(f"T = {w[:-1]} * outlier_p_i + {w[-1]}")
        optim_value = params.fun
        self.w = w

        return self.get_optim_params()

    def calibrate(self, logits, outlier_p):
        w = self.w
        t = self.get_temperature(w, outlier_p)
        return np_softmax(logits / t[:, None])

    def calibrate_before_softmax(self, logits, outlier_p):
        w = self.w
        t = self.get_temperature(w, outlier_p)
        return logits / t[:, None]

    def get_optim_params(self):
        # print(f"T = {self.w} * outlier_p")
        return self.w

