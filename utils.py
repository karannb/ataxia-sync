import os
import random
import itertools
import numpy as np
import pandas as pd
    

def splitter(length) -> tuple:
    if isinstance(length, int):
        inds = np.arange(length)
        np.random.shuffle(inds)
        train2test = int(length * 0.8)
        return inds[:train2test], inds[train2test:]
    elif isinstance(length, list) or isinstance(length, np.ndarray):
        train2test = int(len(length) * 0.8)
        random.shuffle(length)
        return length[:train2test], length[train2test:]
    else:
        raise NotImplementedError
    
def get_10_folds(train_val_inds):
    if isinstance(train_val_inds, list):
        train_inds = []
        val_inds = []
        total = len(train_val_inds)
        random.shuffle(train_val_inds)
        for i in range(1, 11):
            start_idx = int(0.1*total*(i-1))
            end_idx = int(0.1*total*i)
            val_inds.append(train_val_inds[start_idx:end_idx])
            train_inds.append(train_val_inds[:start_idx] + train_val_inds[end_idx:])
        return train_inds, val_inds
    else:
        raise NotImplementedError
    

def train_val_test_split_inds(df : pd.DataFrame):
    
    vids_plus = [int(x) for x in os.listdir("data/gait_cycles/") if \
        (df[df["video"] == int(x)]["label"].values[0] == 1)]
    vids_minus = [int(x) for x in os.listdir("data/gait_cycles/") if \
        (df[df["video"] == int(x)]["label"].values[0] == 0)]
    
    holdout = random.sample(vids_plus, k=5)
    holdout.extend(random.sample(vids_minus, k=5))
    test_inds = list(itertools.chain(*[df[df["video"] == i]["index"].tolist() for i in holdout]))
    vids = list(set(range(len(df))) - set(test_inds))
    train_inds, val_inds = get_10_folds(vids)
    
    return train_inds, val_inds, test_inds

if __name__ == '__main__':
    
    data = pd.read_csv("data/all_gait.csv", index_col=None)
    train, val, test = train_val_test_split_inds(data)
    print(len(train[0]), len(val[0]), len(test))