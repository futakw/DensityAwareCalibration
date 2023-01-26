# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Imported and modified from:

@author: zhang64
"""


import torch
import numpy as np
import torch.nn.parallel

from KDEpy import FFTKDE


def mirror_1d(d, xmin=None, xmax=None):
    """If necessary apply reflecting boundary conditions."""
    if xmin is not None and xmax is not None:
        xmed = (xmin + xmax) / 2
        return np.concatenate(
            (
                (2 * xmin - d[d < xmed]).reshape(-1, 1),
                d,
                (2 * xmax - d[d >= xmed]).reshape(-1, 1),
            )
        )
    elif xmin is not None:
        return np.concatenate((2 * xmin - d, d))
    elif xmax is not None:
        return np.concatenate((d, 2 * xmax - d))
    else:
        return d


def ece_kde_binary(p, label, p_int=None, order=1):

    # points from numerical integration
    if p_int is None:
        p_int = np.copy(p)

    p = np.clip(p, 1e-256, 1 - 1e-256)
    p_int = np.clip(p_int, 1e-256, 1 - 1e-256)

    x_int = np.linspace(-0.6, 1.6, num=2**14)  # x points to use after KDE estimated.

    N = p.shape[0]

    # this is needed to convert labels from one-hot to conventional form
    label_index = np.array([np.where(r == 1)[0][0] for r in label])
    with torch.no_grad():
        if p.shape[1] != 2:  # if multiclass n > 2
            p_new = torch.from_numpy(p)
            p_b = torch.zeros(N, 1)  # softmax
            label_binary = np.zeros((N, 1))
            for i in range(N):
                pred_label = int(torch.argmax(p_new[i]).numpy())
                if pred_label == label_index[i]:
                    label_binary[i] = 1
                p_b[i] = p_new[i, pred_label] / torch.sum(p_new[i, :])
        else:  # if binary classification
            p_b = torch.from_numpy((p / np.sum(p, 1)[:, None])[:, 1])
            label_binary = label_index

    method = "triweight"

    # PP1: p(z) ... estimated density of confidence being z
    dconf_1 = (
        p_b[np.where(label_binary == 1)].reshape(-1, 1)
    ).numpy()  # Confidences of correct preds. Incorrect ones are useless to predict p(z).
    # print( np.std(dconf_1))
    # determine boundwidth: ??????
    kbw = np.std(p_b.numpy()) * (N * 2) ** -0.2  # <= should be deleted??
    kbw = np.std(dconf_1) * (N * 2) ** -0.2
    # Mirror the data about the domain boundary
    low_bound = 0.0
    up_bound = 1.0
    dconf_1m = mirror_1d(dconf_1, low_bound, up_bound)  # ???????????
    # print("kde: dconf_1, dconf_1m:",dconf_1, dconf_1m)
    # Compute KDE using the bandwidth found, and twice as many grid points
    if kbw > 0:
        pp1 = FFTKDE(bw=kbw, kernel=method).fit(dconf_1m).evaluate(x_int)
    else:
        kbw = 0.001
        pp1 = FFTKDE(bw=kbw, kernel=method).fit(dconf_1m).evaluate(x_int)
        print(
            "wrong kbw:",
            kbw,
        )
        print(
            "dconf_1:",
            dconf_1,
        )
        # sys.exit()
    pp1[x_int <= low_bound] = 0  # Set the KDE to zero outside of the domain
    pp1[x_int >= up_bound] = 0  # Set the KDE to zero outside of the domain
    pp1 = pp1 * 2  # Double the y-values to get integral of ~1

    p_int = p_int / np.sum(p_int, 1)[:, None]
    N1 = p_int.shape[0]
    with torch.no_grad():
        p_new = torch.from_numpy(p_int)
        pred_b_int = np.zeros((N1, 1))
        if p_int.shape[1] != 2:
            for i in range(N1):
                pred_label = int(torch.argmax(p_new[i]).numpy())
                pred_b_int[i] = p_int[i, pred_label]
        else:
            for i in range(N1):
                pred_b_int[i] = p_int[i, 1]

    # PP2: p(z) ... Estimated density of conf being z, only using confs of integration points
    low_bound = 0.0
    up_bound = 1.0
    pred_b_intm = mirror_1d(pred_b_int, low_bound, up_bound)
    # Compute KDE using the bandwidth found, and twice as many grid points
    pp2 = FFTKDE(bw=kbw, kernel=method).fit(pred_b_intm).evaluate(x_int)
    pp2[x_int <= low_bound] = 0  # Set the KDE to zero outside of the domain
    pp2[x_int >= up_bound] = 0  # Set the KDE to zero outside of the domain
    pp2 = pp2 * 2  # Double the y-values to get integral of ~1

    # print(len(pp2), (pp1 == pp2).sum())

    if p.shape[1] != 2:  # top label (confidence)
        perc = np.mean(label_binary)
    else:  # or joint calibration for binary cases
        perc = np.mean(label_index)

    integral = np.zeros(x_int.shape)
    reliability = np.zeros(x_int.shape)
    for i in range(x_int.shape[0]):
        conf = x_int[i]  # x point
        conf_i = np.abs(x_int - conf).argmin()  # idx of the x point
        if np.max([pp1[conf_i], pp2[conf_i]]) > 1e-6:
            accu = np.min([perc * pp1[conf_i] / pp2[conf_i], 1.0])
            # if np.max([pp1[np.abs(x_int-conf).argmin()],pp2[np.abs(x_int-conf).argmin()]])>1e-6:
            #     accu = np.min([perc*pp1[np.abs(x_int-conf).argmin()]/pp2[np.abs(x_int-conf).argmin()],1.0])
            if np.isnan(accu) == False:
                integral[i] = np.abs(conf - accu) ** order * pp2[i]
                reliability[i] = accu
        else:
            if i > 1:
                integral[i] = integral[i - 1]

    ind = np.where((x_int >= 0.0) & (x_int <= 1.0))
    # print(np.trapz(pp2[ind],x_int[ind]))
    return np.trapz(integral[ind], x_int[ind]) / np.trapz(pp2[ind], x_int[ind])



def ece_kde_binary_from_conf_acc(confidences, accuracies, p_int=None, order=1):

    # # points from numerical integration
    # if p_int is None:
    #     p_int = np.copy(p)
    N = confidences.shape[0]

    x_int = np.linspace(-0.6, 1.6, num=2**14)  # x points to use after KDE estimated.

    # conf to tensor
    p_b = torch.from_numpy(confidences)
    label_binary = accuracies

    # points from numerical integration
    if p_int is None:
        pred_b_int = np.copy(p_b).reshape(-1, 1)
        

    method = "triweight"

    # PP1: p(z) ... estimated density of confidence being z
    dconf_1 = (
        p_b[np.where(label_binary == 1)].reshape(-1, 1)
    ).numpy()  # Confidences of correct preds. Incorrect ones are useless to predict p(z).
    # print( np.std(dconf_1))
    # determine boundwidth: ??????
    kbw = np.std(p_b.numpy()) * (N * 2) ** -0.2  # <= should be deleted??
    kbw = np.std(dconf_1) * (N * 2) ** -0.2
    # Mirror the data about the domain boundary
    low_bound = 0.0
    up_bound = 1.0
    dconf_1m = mirror_1d(dconf_1, low_bound, up_bound)  # ???????????
    # print(dconf_1, dconf_1m)
    # Compute KDE using the bandwidth found, and twice as many grid points
    if kbw > 0:
        pp1 = FFTKDE(bw=kbw, kernel=method).fit(dconf_1m).evaluate(x_int)
    else:
        print(
            "wrong kbw:",
            kbw,
        )
        print(
            "dconf_1:",
            dconf_1,
        )
        sys.exit()
    pp1[x_int <= low_bound] = 0  # Set the KDE to zero outside of the domain
    pp1[x_int >= up_bound] = 0  # Set the KDE to zero outside of the domain
    pp1 = pp1 * 2  # Double the y-values to get integral of ~1

    # PP2: p(z) ... Estimated density of conf being z, only using confs of integration points
    low_bound = 0.0
    up_bound = 1.0
    pred_b_intm = mirror_1d(pred_b_int, low_bound, up_bound)
    # Compute KDE using the bandwidth found, and twice as many grid points
    pp2 = FFTKDE(bw=kbw, kernel=method).fit(pred_b_intm).evaluate(x_int)
    pp2[x_int <= low_bound] = 0  # Set the KDE to zero outside of the domain
    pp2[x_int >= up_bound] = 0  # Set the KDE to zero outside of the domain
    pp2 = pp2 * 2  # Double the y-values to get integral of ~1

    # print(len(pp2), (pp1 == pp2).sum())

    # if p.shape[1] != 2:  # top label (confidence)
    #     perc = np.mean(label_binary)
    # else:  # or joint calibration for binary cases
    #     perc = np.mean(label_index)
    perc = np.mean(label_binary)

    integral = np.zeros(x_int.shape)
    reliability = np.zeros(x_int.shape)
    for i in range(x_int.shape[0]):
        conf = x_int[i]  # x point
        conf_i = np.abs(x_int - conf).argmin()  # idx of the x point
        if np.max([pp1[conf_i], pp2[conf_i]]) > 1e-6:
            accu = np.min([perc * pp1[conf_i] / pp2[conf_i], 1.0])
            # if np.max([pp1[np.abs(x_int-conf).argmin()],pp2[np.abs(x_int-conf).argmin()]])>1e-6:
            #     accu = np.min([perc*pp1[np.abs(x_int-conf).argmin()]/pp2[np.abs(x_int-conf).argmin()],1.0])
            if np.isnan(accu) == False:
                integral[i] = np.abs(conf - accu) ** order * pp2[i]
                reliability[i] = accu
        else:
            if i > 1:
                integral[i] = integral[i - 1]

    ind = np.where((x_int >= 0.0) & (x_int <= 1.0))
    # print(np.trapz(pp2[ind],x_int[ind]))
    return np.trapz(integral[ind], x_int[ind]) / np.trapz(pp2[ind], x_int[ind])



def ece_hist_binary(p, label, n_bins=15, order=1):
    # binary: correct or incorrect?
    # label: one-hot

    p = np.clip(p, 1e-256, 1 - 1e-256)  # 0 to 1, why?????

    N = p.shape[0]  # the number of data
    label_index = np.array([np.where(r == 1)[0][0] for r in label])  # one hot to index
    with torch.no_grad():
        if p.shape[1] != 2:  # not binary classfication
            preds_new = torch.from_numpy(p)  # just convert into tensor
            preds_b = torch.zeros(N, 1)
            label_binary = np.zeros((N, 1))  # Prediction, Correct or Wrong
            for i in range(N):
                pred_label = int(torch.argmax(preds_new[i]).numpy())
                if pred_label == label_index[i]:
                    label_binary[i] = 1
                # why?????
                preds_b[i] = preds_new[i, pred_label] / torch.sum(preds_new[i, :])
        else:  # if binary classification
            preds_b = torch.from_numpy((p / np.sum(p, 1)[:, None])[:, 1])
            label_binary = label_index

        confidences = preds_b
        accuracies = torch.from_numpy(label_binary)

        x = confidences.numpy()
        x = np.sort(x, axis=0)

        # GET bin boundries
        binCount = int(len(x) / n_bins)  # number of data points in each bin
        bins = np.zeros(n_bins)  # initialize the bins values
        for i in range(0, n_bins, 1):
            bins[i] = x[
                min((i + 1) * binCount, x.shape[0] - 1)
            ]  # max confidence of each bin
            # print((i+1) * binCount)
        bin_boundaries = torch.zeros(len(bins) + 1, 1)
        bin_boundaries[1:] = torch.from_numpy(bins).reshape(-1, 1)
        bin_boundaries[0] = 0.0
        bin_boundaries[-1] = 1.0
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece_avg = torch.zeros(1)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(
                bin_upper.item()
            )  # in_bin: indexes of confidences in the bin
            prop_in_bin = in_bin.float().mean()  # (samples in the bin)/(all samples)
            # print(prop_in_bin)
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece_avg += (
                    torch.abs(avg_confidence_in_bin - accuracy_in_bin) ** order
                    * prop_in_bin
                )
    return ece_avg.cpu().numpy()[0]



def ece_hist_binary_from_conf_acc(confidences, accuracies, n_bins=15, order=1):
    N = confidences.shape[0]

    # to tensor
    confidences = torch.from_numpy(confidences)
    accuracies = torch.from_numpy(accuracies)

    # binary: correct or incorrect?

    x = confidences.numpy()
    x = np.sort(x, axis=0)

    # GET bin boundries
    binCount = int(len(x) / n_bins)  # number of data points in each bin
    bins = np.zeros(n_bins)  # initialize the bins values
    for i in range(0, n_bins, 1):
        bins[i] = x[
            min((i + 1) * binCount, x.shape[0] - 1)
        ]  # max confidence of each bin
        # print((i+1) * binCount)
    bin_boundaries = torch.zeros(len(bins) + 1, 1)
    bin_boundaries[1:] = torch.from_numpy(bins).reshape(-1, 1)
    bin_boundaries[0] = 0.0
    bin_boundaries[-1] = 1.0
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece_avg = torch.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(
            bin_upper.item()
        )  # in_bin: indexes of confidences in the bin
        prop_in_bin = in_bin.float().mean()  # (samples in the bin)/(all samples)
        # print(prop_in_bin)
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece_avg += (
                torch.abs(avg_confidence_in_bin - accuracy_in_bin) ** order
                * prop_in_bin
            )
    return ece_avg.cpu().numpy()[0]


def ece_binary(p, label, n_bins=15, order=1, get_data=False):
    """
    if len(p.shape) == 1: added
    """
    # binary: correct or incorrect?

    p = np.clip(p, 1e-256, 1 - 1e-256)

    N = p.shape[0]  # the number of data
    label_index = np.array([np.where(r == 1)[0][0] for r in label])  # one hot to index
    with torch.no_grad():
        if p.shape[1] != 2:  # not binary classfication
            preds_new = torch.from_numpy(p)  # just convert into tensor
            preds_b = torch.zeros(N, 1)
            label_binary = np.zeros((N, 1))  # Prediction, Correct or Wrong
            for i in range(N):
                pred_label = int(torch.argmax(preds_new[i]).numpy())
                if pred_label == label_index[i]:
                    label_binary[i] = 1
                # why?????
                preds_b[i] = preds_new[i, pred_label] / torch.sum(preds_new[i, :])
        else:  # if binary classification
            preds_b = torch.from_numpy((p / np.sum(p, 1)[:, None])[:, 1])
            label_binary = label_index

        confidences = preds_b
        accuracies = torch.from_numpy(label_binary)

        x = confidences.numpy()
        x = np.sort(x, axis=0)

        ece_avg = torch.zeros(1)
        acc_in_bin = torch.zeros(n_bins)
        conf_in_bin = torch.zeros(n_bins)
        n_in_bin = torch.zeros(n_bins)
        gap = 1.0 / n_bins
        for i in range(n_bins):
            low = i * gap
            high = (i + 1) * gap
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(low) * confidences.le(
                high
            )  # in_bin: indexes of confidences in the bin
            prop_in_bin = in_bin.float().mean()  # (samples in the bin)/(all samples)
            # print(prop_in_bin)
            if prop_in_bin.item() > 0:
                acc_in_bin[i] = accuracies[in_bin].float().mean()
                conf_in_bin[i] = confidences[in_bin].mean()
                n_in_bin[i] = (in_bin == True).sum()
                ece_avg += (
                    torch.abs(conf_in_bin[i] - acc_in_bin[i]) ** order * prop_in_bin
                )

        ece_avg = ece_avg.cpu().numpy()[0]

    avg_conf = confidences.mean().numpy()
    avg_acc = accuracies.mean().numpy()

    acc_in_bin, conf_in_bin, prob_in_bin = (
        acc_in_bin.numpy(),
        conf_in_bin.numpy(),
        n_in_bin.numpy() / N,
    )

    d = {
        "n_bins": n_bins,
        "avg_conf": avg_conf,
        "avg_acc": avg_acc,
        "acc_in_bin": acc_in_bin,
        "conf_in_bin": conf_in_bin,
        "prob_in_bin": prob_in_bin,
    }
    if get_data:
        return ece_avg, d
    else:
        return ece_avg


def ece_binary_from_conf_acc(confidences, accuracies, n_bins=15, order=1, get_data=False):
    N = confidences.shape[0]

    # print(confidences)
    # print(accuracies)

    # to tensor
    confidences = torch.from_numpy(confidences).float()
    accuracies = torch.from_numpy(accuracies).float()

    # binary: correct or incorrect?
    x = confidences.numpy()
    x = np.sort(x, axis=0)

    N = x.shape[0]

    ece_avg = torch.zeros(1)
    acc_in_bin = torch.zeros(n_bins)
    conf_in_bin = torch.zeros(n_bins)
    n_in_bin = torch.zeros(n_bins)
    gap = 1.0 / n_bins
    for i in range(n_bins):
        low = i * gap
        high = (i + 1) * gap
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(low) * confidences.le(
            high
        )  # in_bin: indexes of confidences in the bin
        prop_in_bin = in_bin.float().mean()  # (samples in the bin)/(all samples)
        # print(prop_in_bin)
        if prop_in_bin.item() > 0:
            acc_in_bin[i] = accuracies[in_bin].float().mean()
            conf_in_bin[i] = confidences[in_bin].mean()
            n_in_bin[i] = (in_bin == True).sum()
            ece_avg += (
                torch.abs(conf_in_bin[i] - acc_in_bin[i]) ** order * prop_in_bin
            )

    ece_avg = ece_avg.cpu().numpy()[0]

    avg_conf = confidences.mean().numpy()
    avg_acc = accuracies.mean().numpy()

    acc_in_bin, conf_in_bin, prob_in_bin = (
        acc_in_bin.numpy(),
        conf_in_bin.numpy(),
        n_in_bin.numpy() / N,
    )

    d = {
        "n_bins": n_bins,
        "avg_conf": avg_conf,
        "avg_acc": avg_acc,
        "acc_in_bin": acc_in_bin,
        "conf_in_bin": conf_in_bin,
        "prob_in_bin": prob_in_bin,
    }
    if get_data:
        return ece_avg, d
    else:
        return ece_avg


# Copyed from IntraOrder...
# The next two functions are copied from Kull etal implementation
# for testing
from sklearn.preprocessing import label_binarize


def binary_ECE(probs, y_true, power=1, bins=15):

    idx = np.digitize(probs, np.linspace(0, 1, bins)) - 1
    bin_func = (
        lambda p, y, idx: (np.abs(np.mean(p[idx]) - np.mean(y[idx])) ** power)
        * np.sum(idx)
        / len(probs)
    )

    ece = 0
    for i in np.unique(idx):
        ece += bin_func(probs, y_true, idx == i)
    return ece


def classwise_ECE(probs, y_true, power=1, bins=15):

    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true), classes=range(probs.shape[1]))

    n_classes = probs.shape[1]

    return np.sum(
        [
            binary_ECE(probs[:, c], y_true[:, c].astype(float), power=power, bins=bins)
            for c in range(n_classes)
        ]
    )


# def ece_classwise(p, labels, n_bins=15, order=1, get_data=False):
#     """
#     p: one hot (B x C)
#     labels: one hot (B x C)
#     """
#     # label_index = np.array([np.where(r==1)[0][0] for r in label]) # one hot to index

#     # N = p.shape[0] # the number of data
#     # B, num_classes = label.shape

#     # ECE_each = np.zeros(num_classes)
#     # d_each = []
#     # for c in range(num_classes):

#     #     # instances of class c
#     #     this_p = p[:, c]
#     #     this_labels = labels.eq(c).float()

#     #     # get confs and accs
#     #     ece, d = ece_binary(this_p, this_labels,
#     #         n_bins=n_bins, order=order, get_data=True)
#     #     ECE_each[c] = ece
#     #     d_each.append(d)

#     # cw_ECE = ECE_each.mean()
#     softmaxes = torch.from_numpy(p)
#     labels = torch.from_numpy(labels)

#     bin_boundaries = torch.linspace(0, 1, n_bins + 1)
#     bin_lowers = bin_boundaries[:-1]
#     bin_uppers = bin_boundaries[1:]

#     num_classes = softmaxes.shape[1]
#     cw_ece = torch.zeros(1)
#     for j in range(num_classes):
#       confidences_j = softmaxes[:,j]
#       ece_j = torch.zeros(1)
#       for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
#         in_bin = confidences_j.gt(bin_lower.item()) * confidences_j.le(bin_upper.item())
#         prop_in_bin = in_bin.float().mean()
#         if prop_in_bin.item() > 0:
#           accuracy_j_in_bin = labels[in_bin].eq(j).float().mean()
#           avg_confidence_j_in_bin = confidences_j[in_bin].mean()
#           ece_j += torch.abs(avg_confidence_j_in_bin - accuracy_j_in_bin) * prop_in_bin
#       cw_ece += ece_j

#     cw_ece = cw_ece.numpy()
#     if get_data:
#         return cw_ece, d_each
#     else:
#         return cw_ece


def ece_eval_binary(p, label, ece_type="hist"):
    """
    logits should be after softmax!
    """
    p = np.clip(p, 1e-20, 1 - 1e-20)
    mse = np.mean(np.sum((p - label) ** 2, 1))  # Mean Square Error
    N = p.shape[0]
    nll = -np.sum(label * np.log(p)) / N  # log_likelihood
    accu = (
        np.sum(
            (np.argmax(p, 1) - np.array([np.where(r == 1)[0][0] for r in label])) == 0
        )
        / p.shape[0]
    )  # Accuracy

    if ece_type == "hist":
        ece = ece_hist_binary(p, label)  # ECE
        # ece = ece_hist_binary(p,label).cpu().numpy() # ECE
    elif ece_type == "kde":
        # or if KDE is used
        ece = ece_kde_binary(p, label)
    else:
        raise NotImplementedError

    return ece, nll, mse, accu


#### https://github.com/kartikgupta-at-anu/spline-calibration
def ensure_numpy(a):
    if not isinstance(a, np.ndarray):
        a = a.numpy()
    return a


def len0(x):
    # Proper len function that REALLY works.
    # It gives the number of indices in first dimension

    # Lists and tuples
    if isinstance(x, list):
        return len(x)

    if isinstance(x, tuple):
        return len(x)

    # Numpy array
    if isinstance(x, np.ndarray):
        return x.shape[0]

    # Other numpy objects have length zero
    if is_numpy_object(x):
        return 0

    # Unindexable objects have length 0
    if x is None:
        return 0
    if isinstance(x, int):
        return 0
    if isinstance(x, float):
        return 0

    # Do not count strings
    if type(x) == type("a"):
        return 0

    return 0


def get_top_results(scores, labels, nn, inclusive=False, return_topn_classid=False):

    # Different if we want to take inclusing scores
    if inclusive:
        return get_top_results_inclusive(scores, labels, nn=nn)

    #  nn should be negative, -1 means top, -2 means second top, etc
    # Get the position of the n-th largest value in each row
    topn = [np.argpartition(score, nn)[nn] for score in scores]
    nthscore = [score[n] for score, n in zip(scores, topn)]
    labs = [1.0 if int(label) == int(n) else 0.0 for label, n in zip(labels, topn)]

    # Change to tensor
    tscores = np.array(nthscore)
    tacc = np.array(labs)

    if return_topn_classid:
        return tscores, tacc, topn
    else:
        return tscores, tacc


def one_hot2indices(labels):
    if len(labels.shape) == 2:
        labels = np.argmax(labels, 1)
    return labels


def KS_error(
    logits,
    labels,
):

    # to indices
    labels = one_hot2indices(labels)

    # get confidences of top 1 class
    n = -1
    scores, labels, scores_class = get_top_results(
        logits, labels, n, return_topn_classid=True
    )

    scores = ensure_numpy(scores)
    labels = ensure_numpy(labels)

    # Sort the data
    order = scores.argsort()
    scores = scores[order]
    labels = labels[order]

    # Accumulate and normalize by dividing by num samples
    nsamples = len0(scores)
    integrated_scores = np.cumsum(scores) / nsamples
    integrated_accuracy = np.cumsum(labels) / nsamples
    # percentile = np.linspace (0.0, 1.0, nsamples)
    # fitted_accuracy, fitted_error = compute_accuracy (scores, labels, spline_method, splines, outdir, plotname, showplots=showplots)

    # Work out the Kolmogorov-Smirnov error
    KS_error_max = np.amax(np.absolute(integrated_scores - integrated_accuracy))

    return KS_error_max


def KS_error_from_conf_acc(
    confidences,
    accuracies,
):

    scores = confidences
    labels = accuracies

    scores = ensure_numpy(scores)
    labels = ensure_numpy(labels)

    # Sort the data
    order = scores.argsort()
    scores = scores[order]
    labels = labels[order]

    # Accumulate and normalize by dividing by num samples
    nsamples = len0(scores)
    integrated_scores = np.cumsum(scores) / nsamples
    integrated_accuracy = np.cumsum(labels) / nsamples
    # percentile = np.linspace (0.0, 1.0, nsamples)
    # fitted_accuracy, fitted_error = compute_accuracy (scores, labels, spline_method, splines, outdir, plotname, showplots=showplots)

    # Work out the Kolmogorov-Smirnov error
    KS_error_max = np.amax(np.absolute(integrated_scores - integrated_accuracy))

    return KS_error_max


def ece_eval_all_from_conf_acc(confidences, accuracies, n_bins=15):
    mse = 999
    nll = 999
    accu = accuracies.mean()

    ece_dict = {}
    ece_1, ece_1_d = ece_binary_from_conf_acc(confidences, accuracies, n_bins=n_bins, order=1, get_data=True)
    ece_dict["ece_1"] = ece_1
    ece_dict["ece_1_d"] = ece_1_d
    ece_dict["ece_hist_1"] = ece_hist_binary_from_conf_acc(confidences, accuracies, n_bins=n_bins, order=1)
    ece_dict["ece_kde_1"] = ece_kde_binary_from_conf_acc(confidences, accuracies, order=1)
    ece_dict["cw_ece_1"] = 999
    ece_dict["KS"] = KS_error_from_conf_acc(confidences, accuracies)
    return ece_dict, nll, mse, accu


def ece_eval_all(p, label, n_bins=15):
    """
    logits should be after softmax! (B, C)
    label: (B, C)
    """
    assert len(label.shape) == 2, label.shape

    p = np.clip(p, 1e-20, 1 - 1e-20)

    mse = np.mean(np.sum((p - label) ** 2, 1))  # Mean Square Error
    N = p.shape[0]
    nll = -np.sum(label * np.log(p)) / N  # log_likelihood
    accu = (
        np.sum(
            (np.argmax(p, 1) - np.array([np.where(r == 1)[0][0] for r in label])) == 0
        )
        / p.shape[0]
    )  # Accuracy

    ece_dict = {}
    ece_1, ece_1_d = ece_binary(p, label, n_bins=n_bins, order=1, get_data=True)
    ece_dict["ece_1"] = ece_1
    ece_dict["ece_1_d"] = ece_1_d
    ece_dict["ece_hist_1"] = ece_hist_binary(p, label, n_bins=n_bins, order=1)
    ece_dict["ece_kde_1"] = ece_kde_binary(p, label, order=1)
    # cw_ece_1, cw_ece_1_d = ece_classwise(p,label,n_bins=n_bins,order=1, get_data=True)
    ece_dict["cw_ece_1"] = classwise_ECE(p, np.argmax(label, 1))
    # ece_dict["cw_ece_1_d"] = cw_ece_1_d

    ece_dict["KS"] = KS_error(p, label)

    return ece_dict, nll, mse, accu


def eval_metrics(p, label, n_bins=15):
    """
    p: probabilities (N, C) numpy array
    label: labels (N, C) numpy array (one hot vector)
    """
    # label should be one hot vector
    N, C = p.shape
    if len(label.shape) == 1:
        label = np.eye(C)[label[:, None]]
    elif label.shape[1] == 1:
        label = np.eye(C)[label]

    ece_1, ece_1_d = ece_binary(p, label, n_bins=n_bins, order=1, get_data=True)
    # cw_ece_1, cw_ece_1_d = ece_classwise(p,label,n_bins=n_bins,order=1, get_data=True)
    cw_ece_1 = classwise_ECE(p, np.argmax(label, 1))

    metrics = {
        "mse": np.mean(np.sum((p - label) ** 2, 1)),
        "nll": -np.sum(label * np.log(p)) / N,
        "acc": (
            np.sum(
                (np.argmax(p, 1) - np.array([np.where(r == 1)[0][0] for r in label]))
                == 0
            )
            / p.shape[0]
        ),
        "ece_1": ece_1,
        "cw_ece_1": cw_ece_1,
        "ece_hist_1": ece_hist_binary(p, label, n_bins=n_bins, order=1),
        "ece_kde_1": ece_kde_binary(p, label, order=1),
        "KS": KS_error(p, label),
    }

    bins_data = {
        "ece_1": ece_1_d,
        # "cw_ece_1": cw_ece_1_d,
    }
    return metrics, bins_data


# class EvalMeter:
#     def __init__(self):
#         self.best_metrics = {
#             "mse": {},
#             "nll": {},
#             "ece_1": {},
#             "cw_ece_1": {},
#             "ece_hist_1": {},
#             "ece_kde_1": {},
#         }


#     def add(self, p, label, val_p, val_label, best_metric="mse"):

#         metrics, bins_data = self.eval(p, label)
#         metrics, bins_data = self.eval(val_p, val_label)


#     def eval(self, p, label, n_bins=15):
#         """
#         p: probabilities (N, C) numpy array
#         label: labels (N, C) numpy array (one hot vector)
#         """
#         # label should be one hot vector
#         N, C = p.shape
#         if len(label.shape) == 1:
#             label = np.eye(C)[label[:,None]]
#         elif label.shape[1] == 1:
#             label = np.eye(C)[label]

#         ece_1, ece_1_d = ece_binary(p,label,n_bins=n_bins,order=1,get_data=True)
#         cw_ece_1, cw_ece_1_d = ece_classwise(p,label,n_bins=n_bins,order=1, get_data=True)

#         metrics = {
#             "mse": np.mean(np.sum((p-label)**2,1)),
#             "nll": -np.sum(label*np.log(p))/N,
#             "acc": (np.sum((np.argmax(p,1)-np.array([np.where(r==1)[0][0] for r in label]))==0)/p.shape[0]),
#             "ece_1": ece_1,
#             "cw_ece_1": cw_ece_1,
#             "ece_hist_1": ece_hist_binary(p,label,n_bins=n_bins,order=1),
#             "ece_kde_1": ece_kde_binary(p,label,order=1),
#         }

#         bins_data = {
#             "ece_1": ece_1_d,
#             "cw_ece_1": cw_ece_1_d,
#         }
#         return metrics, bins_data
