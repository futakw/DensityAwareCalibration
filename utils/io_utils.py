import pickle
import numpy as np


def save_pickle(p, data, verbose=0):
    with open(p, "wb") as f:
        pickle.dump(data, f)
    if verbose > 0:
        print(f"\nSaved: {p}")


def load_pickle(p, verbose=0):
    with open(p, "rb") as f:
        data = pickle.load(f)
    if verbose > 0:
        print(f"\Loaded: {p}")
    return data


def save_numpy(p, data, verbose=0):
    with open(p, "wb") as f:
        np.save(f, data)
    if verbose > 0:
        print(f"\nSaved: {p}")


def load_numpy(p, verbose=0):
    with open(p, "rb") as f:
        data = np.load(f)
    return data
