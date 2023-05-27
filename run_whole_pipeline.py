import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate

from utils.utils import softmax, calc_acc
from utils.io_utils import load_pickle, save_pickle
from utils.util_calibration import (
    ts_calibrate,
    ets_calibrate,
    mir_calibrate,
    irova_calibrate,
)
from utils.util_evaluation import ece_eval_all, ece_eval_all_from_conf_acc
from utils.spline import get_spline_calib_func, spline_calibrate

from density_aware_calib import KNNScorer, DAC

###########################################
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("--save_outputs_dir", type=str, default="")
parser.add_argument("-d", "--dataset", type=str, default="cifar10")
parser.add_argument("-c", "--num_classes", type=int, default=10)
parser.add_argument("-m", "--model_name", type=str, default="resnet18")
parser.add_argument("--ood_vector_type", type=str, default="multi")
# knn_k: CIFAR10: 50, CIFAR100: 200, ImageNet(1%): 10
parser.add_argument("--knn_k", type=int, default=50)
parser.add_argument("--ood_values_num", type=int, default=6) 
parser.add_argument(
    "--ood_scoring_layers_list",
    type=str,
    nargs="*",
    default=["maxpool", "layer1", "layer2", "layer3", "layer4", "logits"],
)
parser.add_argument(
    "--combination_method", type=str, default="ETS", choices=["ETS", "SPL"]
)
parser.add_argument(
    "--test_data_type", type=str, nargs="*", default="gaussian_noise_3"
)

args = parser.parse_args()

if args.dataset == "cifar10":
    args.num_classes = 10
    args.knn_k = 50
elif args.dataset == "cifar100":
    args.num_classes = 100
    args.knn_k = 200
elif args.dataset == "imagenet":
    args.num_classes = 1000
    args.knn_k = 10 # if using 1% of ImageNet data
else:
    raise NotImplementedError

args.ood_values_num = len(args.ood_scoring_layers_list)
test_data_type = args.test_data_type

########### get data loader#############
print("\nGet data loader.")
train_loader, val_loader, test_loader = get_loaders(
    name=args.dataset,
    batch_size=args.batch_size,
    train_no_aug=True,  # important
)

###### Prepare KNN score calculator ######
print("\nPrepare KNN score calculator.")
train_save_d = os.path.join(args.save_outputs_dir, "train")
train_labels = load_pickle(os.path.join(train_save_d, "labels.pickle"))
train_outputs = load_pickle(os.path.join(train_save_d, "outputs.pickle"))

ood_scorer = KNNScorer(
    top_k=args.knn_k,
)

# load features and set them to the KNNscorer
for layer in args.ood_scoring_layers_list:
    print(" - ", layer)
    train_feat = load_pickle(
        os.path.join(train_save_d, "features", f"{layer}.pickle")
    )
    ood_scorer.set_train_feat(train_feat, train_labels, args.num_classes)

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

val_ood_scores = ood_scorer.get_score(val_feat)

# optimize DAC
DAC_calibrator = DAC(ood_values_num=args.ood_values_num)
optim_params = DAC_calibrator.optimize(
    val_outputs, val_labels, val_ood_scores, loss="mse"
)

#########  Inference ###########
results = []

# calibrate on validation set
print("\n------------------\nCalibrate on validation set.")
val_calib_logits = DAC_calibrator.calibrate_before_softmax(
    val_outputs, val_ood_scores
)
if args.combination_method == "ETS":
    # without DAC
    p_wo_DAC = ets_calibrate(val_outputs, val_labels, val_outputs, args.num_classes, "mse")
    # with DAC
    p = ets_calibrate(val_calib_logits, val_labels, val_calib_logits, args.num_classes, "mse")
elif args.combination_method == "SPL":
    # without DAC
    SPL_frecal, p_wo_DAC, label_wo_DAC = get_spline_calib_func(val_outputs, val_labels)
    # with DAC
    SPL_DAC_frecal, p, label = get_spline_calib_func(val_calib_logits, val_labels)
else:
    raise NotImplementedError

print("\n-------------------\nCalibration performance on validation set")
if args.combination_method != "SPL":
    # without DAC
    test_ece_dict, test_nll, test_mse, test_accu = ece_eval_all(p_wo_DAC, val_labels)
    test_ece_1_wo_DAC = test_ece_dict["ece_1"]
    # with DAC
    test_ece_dict, test_nll, test_mse, test_accu = ece_eval_all(p, val_labels)
    test_ece_1_with_DAC = test_ece_dict["ece_1"]
else:
    # Spline only calibrates top 1 prediction, so use different func. to evaluate
    # without DAC
    test_ece_dict, test_nll, test_mse, test_accu = ece_eval_all_from_conf_acc(p_wo_DAC, label_wo_DAC)
    test_ece_1_wo_DAC = test_ece_dict["ece_1"]
    # with DAC
    test_ece_dict, test_nll, test_mse, test_accu = ece_eval_all_from_conf_acc(p, label)
    test_ece_1_with_DAC = test_ece_dict["ece_1"]
print(f"- {args.combination_method} w/o DAC: test_ece_1:", test_ece_1_wo_DAC)
print(f"- {args.combination_method} + DAC: test_ece_1:", test_ece_1_with_DAC)
results.append(["val", test_ece_1_wo_DAC, test_ece_1_with_DAC])

# calibrate test set
for k in test_data_type:
    print("\n-------------------\nCalibrate on test set: ", k)

    # calc KNN score
    this_save_d = os.path.join(args.save_outputs_dir, k)
    this_labels = load_pickle(os.path.join(this_save_d, "labels.pickle"))
    this_outputs = load_pickle(os.path.join(this_save_d, "outputs.pickle"))
    acc = calc_acc(this_outputs, this_labels)
    print(f"Test Acc ({k}): ", acc)

    print(f"Calculating knn scores on test set ({k})...")
    this_feat = []
    for layer in args.ood_scoring_layers_list:
        path = os.path.join(this_save_d, "features", f"{layer}.pickle")
        this_feat.append(load_pickle(path))

    # get OOD scores
    this_ood_scores = ood_scorer.get_score(this_feat)

    # calibrate
    print(f"Calibrating ({k})...")
    calib_logits_eval = DAC_calibrator.calibrate_before_softmax(
        this_outputs, this_ood_scores
    )
    if args.combination_method == "ETS":
        p_eval_wo_DAC = ets_calibrate(
            val_outputs, val_labels, this_outputs, args.num_classes, "mse"
        )
        p_eval = ets_calibrate(
            val_calib_logits, val_labels, calib_logits_eval, args.num_classes, "mse"
        )
    elif args.combination_method == "SPL":
        p_eval_wo_DAC, label_eval_wo_DAC = spline_calibrate(
            SPL_frecal, this_outputs, this_labels
        )
        p_eval, label_eval = spline_calibrate(
            SPL_DAC_frecal, calib_logits_eval, this_labels
        )
    else:
        raise NotImplementedError

    # evaluate
    print(f"\nCalibration performance on test set: {k}")

    if args.combination_method != "SPL":
        # without DAC
        test_ece_dict, test_nll, test_mse, test_accu = ece_eval_all(p_eval_wo_DAC, this_labels)
        test_ece_1_wo_DAC = test_ece_dict["ece_1"]
        # with DAC
        test_ece_dict, test_nll, test_mse, test_accu = ece_eval_all(p_eval, this_labels)
        test_ece_1_with_DAC = test_ece_dict["ece_1"]
    else:
        # Spline only calibrates top 1 prediction, so use different func. to evaluate
        # without DAC
        test_ece_dict, test_nll, test_mse, test_accu = ece_eval_all_from_conf_acc(p_eval_wo_DAC, label_eval_wo_DAC)
        test_ece_1_wo_DAC = test_ece_dict["ece_1"]
        # with DAC
        test_ece_dict, test_nll, test_mse, test_accu = ece_eval_all_from_conf_acc(p_eval, this_labels)
        test_ece_1_with_DAC = test_ece_dict["ece_1"]
    print(f"- {args.combination_method} w/o DAC: test_ece_1:", test_ece_1_wo_DAC)
    print(f"- {args.combination_method} + DAC: test_ece_1:", test_ece_1_with_DAC)
    results.append([k, test_ece_1_wo_DAC, test_ece_1_with_DAC])


df = pd.DataFrame(
    data=results, 
    columns=["test data", f"{args.combination_method} w/o DAC", f"{args.combination_method} + DAC"]
)
print(tabulate(df, headers='keys', tablefmt='psql'))