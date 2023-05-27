"""
To save outputs (features or OOD scores) output by a classifier.
"""
import os
import glob
import sys
import numpy as np
import pickle
from tqdm import tqdm
import time
import copy
import gc
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor

import faiss
from utils.io_utils import save_pickle, load_pickle
from utils.get_models import FeatureExtractor
from density_aware_calib import KNNScorer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
print("Visible devices num: ", torch.cuda.device_count())


def extract_features_and_save_ood_scores(
    args,
    split,
    loader,
    feature_extractor,
    to_save_features=False,
    ood_scorer=None,
    save_batch_interval=50,
):
    """
    1. For training data, ood_scorer should be None, to_save_features should be True.
        Then, extracted features and labels are saved.
    2. For test data, ood_scorer should be given, to_save_features should be False.
        Then, extracted features are not saved, but the OOD scores are saved.
    
    Inputs:
        args: 
            arguments
        split: 
            "train" or "test" or other data name
        loader: 
            data loader
        feature_extractor:  
            the classifier we want to calibrate. (create_feature_extractor())
        save_batch_interval: 
            In order to save memory, we save the outputs in batches.
            Decrease this number if memory error occurs.
    """
    print("Split: ", split)
    # save them
    save_d = os.path.join(args.save_outputs_dir, split)
    save_outputs_d = os.path.join(save_d, "outputs")
    save_labels_d = os.path.join(save_d, "labels")
    save_features_d = os.path.join(save_d, "features")
    os.makedirs(save_d, exist_ok=True)
    os.makedirs(save_outputs_d, exist_ok=True)
    os.makedirs(save_labels_d, exist_ok=True)
    if to_save_features:
        os.makedirs(save_features_d, exist_ok=True)

    if os.path.exists(os.path.join(save_d, f"outputs.pickle")):
        print(f"Already extracted features: ", split)
        return

    # eval mode
    feature_extractor.eval()

    ##############################
    # get features
    batch_sum = len(loader)
    print("total batch size:", batch_sum)

    save_idx = 0
    data = {
        "features": {},
        "outputs": [],
        "labels": [],
    }
    ood_score_dict = {}
    with torch.no_grad():
        for batch_idx, (inputs, labels) in tqdm(enumerate(loader)):
            inputs, labels = inputs.to(device), labels.to(torch.long)

            # get outputs
            features_dict = feature_extractor(inputs)

            for k, feat in features_dict.items():
                feat = feat.detach().cpu().numpy()
                data["features"].setdefault(k, []).append(feat)
                assert len(feat.shape) == 2, feat.shape
                if batch_idx == 0:
                    print(k, feat.shape)

            outputs = features_dict[list(features_dict.keys())[-1]].detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            data["outputs"].append(outputs)
            data["labels"].append(labels)

            del inputs, outputs, labels, feat
            gc.collect()

            to_end = batch_sum * args.end_ratio < batch_idx if split == "train" else False
            if (
                (len(data["outputs"]) >= save_batch_interval)
                or (batch_sum == batch_idx + 1)
                or to_end
            ):
                # concat numpy array
                data["outputs"] = np.concatenate(data["outputs"], 0)
                data["labels"] = np.concatenate(data["labels"], 0)
                for k in data["features"]:
                    data["features"][k] = np.concatenate(data["features"][k], 0)

                # calc OOD score
                if ood_scorer is not None:
                    test_feats = [data["features"][k] for k in data["features"]]
                    ood_scores = ood_scorer.get_score(test_feats)
                    for i, k in enumerate(data["features"]):
                        ood_score_dict.setdefault(k, []).append(ood_scores[i])

                # save them
                p = os.path.join(save_outputs_d, f"batch{save_idx}.pickle")
                save_pickle(p, data["outputs"])

                p = os.path.join(save_labels_d, f"batch{save_idx}.pickle")
                save_pickle(p, data["labels"])

                if to_save_features:
                    for k in data["features"]:
                        this_d = os.path.join(save_features_d, k)
                        os.makedirs(this_d, exist_ok=True)
                        p = os.path.join(this_d, f"batch{save_idx}.pickle")
                        save_pickle(p, data["features"][k])

                # reset
                del data
                gc.collect()
                data = {
                    "features": {},
                    "outputs": [],
                    "labels": [],
                }
                print(f"batch: {batch_idx}, save_idx: {save_idx}")
                save_idx += 1

            if to_end:
                print(f"End at data ratio={args.end_ratio}. batch_idx={batch_idx}")
                break

    # combine saved files
    combine_batch_data(args, split)

    if ood_scorer is not None:
        ood_score_dict = {k: np.concatenate(v, 0) for k, v in ood_score_dict.items()}
        save_ood_values(args, split, ood_score_dict)


def combine_batch_data(args, split):
    """
    Combine saved batch data into one file.
    """
    print("Split: ", split)
    # save them
    save_d = os.path.join(args.save_outputs_dir, split)
    os.makedirs(save_d, exist_ok=True)
    save_outputs_d = os.path.join(save_d, "outputs")
    save_labels_d = os.path.join(save_d, "labels")
    save_features_d = os.path.join(save_d, "features")

    if os.path.exists(os.path.join(save_d, "outputs.pickle")):
        print("outputs.pickle already exist.")
    else:
        outputs = []
        i = 0
        while True:
            if not os.path.exists(os.path.join(save_outputs_d, f"batch{i}.pickle")):
                break
            this_outputs = load_pickle(os.path.join(save_outputs_d, f"batch{i}.pickle"))
            outputs.append(this_outputs)
            i += 1
        outputs = np.concatenate(outputs, 0)
        save_pickle(os.path.join(save_d, "outputs.pickle"), outputs)
        del outputs, this_outputs
        gc.collect()
    shutil.rmtree(save_outputs_d, ignore_errors=True)

    if os.path.exists(os.path.join(save_d, "labels.pickle")):
        print("labels.pickle already exist.")
    else:
        labels = []
        i = 0
        while True:
            if not os.path.exists(os.path.join(save_labels_d, f"batch{i}.pickle")):
                break
            this_labels = load_pickle(os.path.join(save_labels_d, f"batch{i}.pickle"))
            labels.append(this_labels)
            i += 1
        labels = np.concatenate(labels, 0)
        save_pickle(os.path.join(save_d, "labels.pickle"), labels)
        del labels, this_labels
        gc.collect()
    shutil.rmtree(save_labels_d, ignore_errors=True)

    outputs = load_pickle(os.path.join(save_d, "outputs.pickle"))
    labels = load_pickle(os.path.join(save_d, "labels.pickle"))
    if len(labels.shape) == 2:
        labels = np.argmax(labels, 1)
    correct = np.argmax(outputs, 1) == labels
    acc = np.array(correct).mean()
    print("Acc: ", acc)
    if split in ["train", "val", "test"] and acc < 0.50:
        print("Something wrong with accuracy!")
        print(outputs)
        print(np.argmax(outputs, 1))
        print(labels)
        exit()

    if (not args.save_only_ood_scores) or (split == "train"):
        feat_list = glob.glob(os.path.join(save_features_d, "*"))
        feat_list = [f.split("/")[-1].replace(".pickle", "") for f in feat_list]
        for k in feat_list:
            this_save_feats_d = os.path.join(save_features_d, k)
            this_feat_save_path = os.path.join(save_features_d, f"{k}.pickle")
            if os.path.exists(this_feat_save_path):
                print(f"{this_feat_save_path} already exist.")
            else:
                feats = []
                i = 0
                while True:
                    this_path = os.path.join(this_save_feats_d, f"batch{i}.pickle")
                    if not os.path.exists(this_path):
                        break
                    this_feats = load_pickle(this_path)
                    feats.append(this_feats)
                    i += 1
                feats = np.concatenate(feats, 0)
                save_pickle(this_feat_save_path, feats)
                del feats, this_feats
                gc.collect()
            shutil.rmtree(this_save_feats_d, ignore_errors=True)


def save_normal_format(args):
    """
    Save outputs and labels in normal format of,
        data = ((logits, labels), (logits_eval, labels_eval))
    """
    save_d = os.path.join(args.save_outputs_dir, "val")
    logits = load_pickle(os.path.join(save_d, "outputs.pickle"))
    labels = load_pickle(os.path.join(save_d, "labels.pickle"))

    save_d = os.path.join(args.save_outputs_dir, "test")
    logits_eval = load_pickle(os.path.join(save_d, "outputs.pickle"))
    labels_eval = load_pickle(os.path.join(save_d, "labels.pickle"))

    data = ((logits, labels), (logits_eval, labels_eval))
    save_p = os.path.join(args.save_outputs_dir, "normal_format.pickle")
    save_pickle(save_p, data)

    return data


def save_ood_values(args, split, data):
    name = "ood_score"
    if args.save_dist_arr:
        name = "ood_score_arr"

    save_d = os.path.join(args.save_outputs_dir, split, name)
    os.makedirs(save_d, exist_ok=True)

    for k in data:
        p = os.path.join(save_d, f"{k}.pickle")
        save_pickle(p, data[k])
        print(f"Saved {k} to {p}: {data[k].shape}")
    print("Saved ood values: ", save_d)


def str2bool(s):
    if s.lower() in ["t", "true"]:
        return True
    elif s.lower() in ["f", "false"]:
        return False
    else:
        raise ValueError


if __name__ == "__main__":
    import sys
    import argparse
    from utils.dataset import get_loaders
    from constants import get_layers_name

    parser = argparse.ArgumentParser(description="")
    # dataset
    parser.add_argument("--data_root", type=str, default="../")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="cifar10")
    # model
    parser.add_argument("--model_name", type=str, default="resnet18")
    # save args
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--is_all_layers", type=str2bool, default=False)
    parser.add_argument("--save_only_ood_scores", type=str2bool, default=True)
    parser.add_argument("--save_dist_arr", type=str2bool, default=False)
    parser.add_argument("--end_ratio", type=float, default=1.0)
    parser.add_argument("--save_batch_interval", type=int, default=100)
    parser.add_argument("--corruption", type=str2bool, default=True)
    parser.add_argument(
        "--test_data_type", type=str, nargs="*", 
        default=["natural", "gaussian_noise"]
    )

    args = parser.parse_args()

    from utils.get_models import *

    if args.dataset == "imagenet":
        args.num_classes = c = 1000
        args.top_k = 10
        args.end_ratio = 0.1
    elif args.dataset == "cifar100":
        args.num_classes = c = 100
        args.top_k = 200
    elif args.dataset == "cifar10":
        args.num_classes = c = 10
        args.top_k = 50
    else:
        raise ValueError

    ###### get data ########
    train_loader, val_loader, test_loader = get_loaders(
        name=args.dataset,
        batch_size=args.batch_size,
        train_no_aug=True,  # important
    )

    args.save_outputs_dir = f"outputs/{args.dataset}/{args.model_name}"
    os.makedirs(args.save_outputs_dir, exist_ok=True)

    # load model
    if "cifar" in args.dataset:
        # load classifier
        model = get_model(
            args.model_name,
            args.num_classes,
        )
        p = f"ckpts/{args.dataset}/{args.model_name}.pth"
        checkpoint = torch.load(p)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        print("loaded model weights: ", p)
    else:
        raise NotImplementedError

    # Feature extractor
    if args.is_all_layers:
        return_nodes = get_layers_name(
            args.model_name, model=model, get_all=True, add_logits=True
        )
        feature_extractor = FeatureExtractor(model, return_nodes)
    else:
        return_nodes = get_layers_name(args.model_name, model=model, add_logits=True)
        feature_extractor = FeatureExtractor(model, return_nodes)
    if isinstance(return_nodes, dict):
        layers_name = [v for k, v in return_nodes.items()]
    else:
        layers_name = return_nodes

    if torch.cuda.device_count() >= 2:
        model = torch.nn.DataParallel(model).cuda()
        feature_extractor = torch.nn.DataParallel(feature_extractor).cuda()

    # train
    start_time = time.time()

    split = "train"
    loader = train_loader
    extract_features_and_save_ood_scores(
        args,
        split,
        loader,
        feature_extractor,
        to_save_features=True,
        ood_scorer=None,
        save_batch_interval=args.save_batch_interval,
    )

    end_time = time.time()
    print(f"- Train feature extraction: {end_time - start_time} seconds")

    # set ood_scorer
    train_save_d = os.path.join(args.save_outputs_dir, "train")
    train_labels = load_pickle(os.path.join(train_save_d, "labels.pickle"))
    ood_scorer = KNNScorer(top_k=args.top_k, return_dist_arr=args.save_dist_arr, gpu=True)
    for layer in layers_name:
        path = os.path.join(train_save_d, "features", f"{layer}.pickle")
        train_feat = load_pickle(path)
        print("Set train features to ood_scorer: {}, {}".format(layer, train_feat.shape))
        ood_scorer.set_train_feat(train_feat, train_labels, args.num_classes)

    # val, test
    for split, loader in zip(["val", "test"], [val_loader, test_loader]):
        print("Save OOD scores: ", split)
        ood_score_dict = extract_features_and_save_ood_scores(
            args,
            split,
            loader,
            feature_extractor,
            to_save_features=False,
            ood_scorer=ood_scorer,
            save_batch_interval=args.save_batch_interval,
        )

    ((logits, labels), (logits_eval, labels_eval)) = save_normal_format(args)

    ###### CORRUPTION DATA
    if args.corruption:
        assert args.dataset in ["cifar10", "cifar100", "imagenet"]

        # test_corruptions = [
        #     "natural",
        #     "gaussian_noise",
        #     "shot_noise",
        #     "speckle_noise",
        #     "impulse_noise",
        #     "defocus_blur",
        #     "gaussian_blur",
        #     "motion_blur",
        #     "zoom_blur",
        #     "snow",
        #     "fog",
        #     "brightness",
        #     "contrast",
        #     "elastic_transform",
        #     "pixelate",
        #     "jpeg_compression",
        #     "spatter",
        #     "saturate",
        #     "frost",
        # ]
        test_corruptions = args.test_data_type
        severities = [1, 2, 3, 4, 5]

        for ci, cname in enumerate(test_corruptions):
            for severity in severities:
                cname_s = f"{cname}_{severity}"
                print("Save OOD scores: ", cname_s)

                loader = get_loaders(
                    f"{args.dataset}c",
                    cname=cname,
                    batch_size=args.batch_size,
                    severity=severity,
                )

                ood_score_dict = extract_features_and_save_ood_scores(
                    args,
                    cname_s,
                    loader,
                    feature_extractor,
                    to_save_features=False,
                    ood_scorer=ood_scorer,
                    save_batch_interval=args.save_batch_interval,
                )

                save_p = os.path.join(
                    args.save_outputs_dir, f"{cname_s}_normal_format.pickle"
                )
                save_d = os.path.join(args.save_outputs_dir, cname_s)
                logits_eval = load_pickle(os.path.join(save_d, "outputs.pickle"))
                labels_eval = load_pickle(os.path.join(save_d, "labels.pickle"))

                data = ((logits, labels), (logits_eval, labels_eval))
                save_pickle(save_p, data)
