import os, glob
import sys
import numpy as np
import math
from tqdm import tqdm
import json
import pickle
import pandas as pd

from utils.util_calibration import ts_calibrate, ets_calibrate, mir_calibrate, irova_calibrate
from utils.util_evaluation import ece_eval_all, ece_eval_all_from_conf_acc

from density_aware_calib import KNNScorer, DAC


def load_pickle(p, verbose=-1):
    with open (p, "rb") as f:
        data = pickle.load(f)
    if verbose > 0:
        print("loaded:", p)
    return data

def save_pickle(p, data, verbose=-1):
    with open(p, "wb") as f:
        pickle.dump(data, f)
    if verbose > 0:
        print("saved:", p)

def softmax(x):
    max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum 
    return f_x

def to_onehot(arr, n_class):
    return np.eye(n_class)[arr]

def calc_acc(outputs_eval, label_eval):
    if len(label_eval.shape) == 2:
        label_eval = np.argmax(label_eval, 1)
    acc = np.array(np.argmax(outputs_eval, 1) == label_eval).mean()
    return acc


if __name__=="__main__":
    def str2bool(s):
        if s.lower() in ["t", "true"]:
            return True
        elif s.lower() in ["f", "false"]:
            return False
        else:
            raise ValueError

    import argparse  

    parser = argparse.ArgumentParser(description='')    
    parser.add_argument('--save_outputs_dir', type=str, default="")  
    parser.add_argument('-d', '--dataset', type=str, default="cifar10")  
    parser.add_argument('-c', '--num_classes', type=int, default=10)  
    parser.add_argument('-m', '--model_name', type=str, default="resnet18")  
    parser.add_argument('--ood_vector_type', type=str, default="multi") 
    parser.add_argument('--knn_k', type=int, default=50)   # CIFAR10: 50, CIFAR100: 200, ImageNet 1%: 10
    parser.add_argument('--ood_values_num', type=int, default=1)  ######
    parser.add_argument('--ood_scoring_layers_list', type=str, nargs="*", 
        default=["maxpool", "layer1", "layer2", "layer3", "layer4", "logits"]) 
    parser.add_argument('--combination_method', 
        type=str, default="ETS", # options=["ETS", "SPL"]
    )

    parser.add_argument('--save_calib_logits', type=str2bool, default=False)
    parser.add_argument('--save_calib_dir', type=str, default="calib_outputs")
    
    parser.add_argument('--test_data_type', type=str, nargs="*", default="gaussian_noise_3")  

    args = parser.parse_args()  


    if args.dataset == "cifar10":
        args.num_classes = 10
    elif args.dataset == "cifar100":
        args.num_classes = 100
    elif args.dataset == "imagenet":
        args.num_classes = 1000
    else:
        raise NotImplementedError

    args.ood_values_num = len(args.ood_scoring_layers_list)
    keys = args.test_data_type

    ###### Prepare KNN score calculator ######
    print("\nPrepare KNN score calculator.")
    train_save_d = os.path.join(args.save_outputs_dir, "train")
    train_labels = load_pickle(os.path.join(train_save_d, "labels.pickle"))
    train_outputs = load_pickle(os.path.join(train_save_d, "outputs.pickle"))

    OODscorer = KNNScorer(
        train_labels,
        top_k=args.knn_k,
    )

    for layer in args.ood_scoring_layers_list:
        print(" - ", layer)
        train_feat = load_pickle(os.path.join(train_save_d, "features", f"{layer}.pickle"))
        OODscorer.set_train_feat(train_feat, train_labels, args.num_classes)
    
    #########  Calibration ###########
    # calculate OOD score for the validation set.
    val_save_d = os.path.join(args.save_outputs_dir, "val")
    val_labels = load_pickle(os.path.join(val_save_d, "labels.pickle"))
    val_outputs = load_pickle(os.path.join(val_save_d, "outputs.pickle"))
    acc = calc_acc(val_outputs, val_labels)
    print("Validation Acc: ", acc)

    print("\nCalculating knn scores on validation set...")
    val_feat = []
    for layer in args.ood_scoring_layers_list:
        val_feat.append(
            load_pickle(os.path.join(val_save_d, "features", f"{layer}.pickle"))
        )
           
    val_outlier_p = OODscorer.get_score(val_outputs, val_feat)

    # optimize DAC
    DAC_calibrator = DAC(ood_values_num=args.ood_values_num)
    optim_params = DAC_calibrator.optimize(
        val_outputs, val_labels, val_outlier_p, loss="mse"
    )


    #########  Inference ###########

    # calibrate validation set
    print("\nCalibrate on validation set.")
    val_calib_logits = DAC_calibrator.calibrate_before_softmax(
        val_outputs, val_outlier_p
    )
    if args.combination_method == "ETS":
        p = ets_calibrate(val_calib_logits, val_labels, 
            val_calib_logits, args.num_classes, 'mse')
    else:
        raise NotImplementedError

    # print("\nCalibration performance of validation set")
    # train_ece_dict, train_nll, train_mse, train_accu = ece_eval_all(p, val_labels)
    # print("=> train_ece_1:", train_ece_dict["ece_1"],)
    # print("=> train_kde_ece:", train_ece_dict["ece_kde_1"],)

    
    # calibrate test set
    for k in keys:

        print("\nCalibrate on test set: ", k)

        # calc KNN score
        this_save_d = os.path.join(args.save_outputs_dir, k)
        this_labels = load_pickle(os.path.join(this_save_d, "labels.pickle"))
        this_outputs = load_pickle(os.path.join(this_save_d, "outputs.pickle"))
        acc = calc_acc(this_outputs, this_labels)
        print(f"Test Acc ({k}): ", acc)

        print(f"Calculating knn scores on test set ({k})...")
        this_feat = []
        for layer in args.ood_scoring_layers_list:
            this_feat.append(
                load_pickle(os.path.join(this_save_d, "features", f"{layer}.pickle"))
            )
            
        this_outlier_p = OODscorer.get_score(this_outputs, this_feat)

        # calibrate
        print(f"Calibrating ({k})...")
        calib_logits_eval = DAC_calibrator.calibrate_before_softmax(
            this_outputs,  this_outlier_p
        )
        if args.combination_method == "ETS":
            p_eval_without_DAC = ets_calibrate(val_outputs, val_labels, 
                this_outputs, args.num_classes, 'mse')
            p_eval = ets_calibrate(val_calib_logits, val_labels, 
                calib_logits_eval, args.num_classes, 'mse')
        else:
            raise NotImplementedError

        # evaluate
        print(f"\nCalibration performance on test set: {k}")
        
        test_ece_dict, test_nll, test_mse, test_accu = ece_eval_all(p_eval_without_DAC, this_labels)
        print(f"- {args.combination_method} w/o DAC: test_ece_1:", test_ece_dict["ece_1"],)
        
        test_ece_dict, test_nll, test_mse, test_accu = ece_eval_all(p_eval, this_labels)
        print(f"- {args.combination_method} + DAC: test_ece_1:", test_ece_dict["ece_1"],)