import numpy as np

def softmax(x):
    max = np.max(
        x, axis=1, keepdims=True
    )  # returns max of each row and keeps same dims
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(
        e_x, axis=1, keepdims=True
    )  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x


def calc_acc(outputs_eval, label_eval):
    if len(label_eval.shape) == 2:
        label_eval = np.argmax(label_eval, 1)
    acc = np.array(np.argmax(outputs_eval, 1) == label_eval).mean()
    return acc


def to_onehot(arr, n_class):
    return np.eye(n_class)[arr]


def label2onehot(label, n_class=None):
    if n_class is None:
        n_class = int(label.max() + 1)
        
    if len(label.shape) == 1:
        label = to_onehot(label, n_class)
    if len(label.shape) == 2 and label.shape[1] == 1:
        label = to_onehot(label[:, 0], n_class)
    return label