import io
import os
import torch
import random
import itertools
import numpy as np
import pandas as pd
from typing import List, Tuple
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def seed_all(seed: int) -> None:
    '''
    Seeds all the random number generators.
    
    Args
    ----
    seed: int
        The seed to be used.
    
    Returns
    -------
    None
    '''
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def splitter(len_or_list) -> Tuple[List[int], List[int]]:
    '''
    NOT USED now.
    Splits the data into training and testing sets.
    '''
    if isinstance(len_or_list, int):
        inds = np.arange(len_or_list)
        np.random.shuffle(inds)
        train2test = int(len_or_list * 0.8)
        return inds[:train2test], inds[train2test:]
    elif isinstance(len_or_list, list) or isinstance(len_or_list, np.ndarray):
        train2test = int(len(len_or_list) * 0.8)
        random.shuffle(len_or_list)
        return len_or_list[:train2test], len_or_list[train2test:]
    else:
        raise NotImplementedError


def get_10_folds(train_val_inds) -> Tuple[List[List[int]], List[List[int]]]:
    '''
    Splits the data into 10 folds for cross-validation.
    
    Args
    ----
    train_val_inds: list
        List of indices to be split into 10 folds.
    
    Returns
    -------
    train_inds: list
        List of lists of indices for training data.
    val_inds: list
        List of lists of indices for validation data.
    '''
    if isinstance(train_val_inds, list):
        train_inds = []
        val_inds = []
        total = len(train_val_inds)
        random.shuffle(train_val_inds)
        for i in range(1, 11):
            start_idx = int(0.1 * total * (i - 1))
            end_idx = int(0.1 * total * i)
            val_inds.append(train_val_inds[start_idx:end_idx])
            train_inds.append(train_val_inds[:start_idx] +
                              train_val_inds[end_idx:])
        return train_inds, val_inds
    else:
        raise NotImplementedError


def train_val_test_split_inds(
        df: pd.DataFrame,
        task: str) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    '''
    Splits the data into training, validation, and testing sets.
    
    Args
    ----
    df: pd.DataFrame
        DataFrame containing the data.
    task: str
        The task being optimized for (classification/regression).
        
    Returns
    -------
    train_inds: list
        List of lists of indices for training data. 10 lists.
    val_inds: list
        List of lists of indices for validation data. 10 lists.
    test_inds: list
        List of indices for testing data.
    '''

    if task == "classification":
        vids_plus = [int(x) for x in os.listdir("data/gait_cycles/") if \
            (df[df["video"] == int(x)]["label"].values[0] == 1)]
        vids_minus = [int(x) for x in os.listdir("data/gait_cycles/") if \
            (df[df["video"] == int(x)]["label"].values[0] == 0)]
        # get the indices of the videos that have label 1 and 0
        # so that I can get a representative sample of the holdout set

        holdout = random.sample(vids_plus, k=5)
        holdout.extend(random.sample(vids_minus, k=5))
    else:
        vids = [int(x) for x in os.listdir("data/gait_cycles/")]
        holdout = random.sample(vids, k=10)

    test_inds = list(
        itertools.chain(
            *[df[df["video"] == i]["index"].tolist() for i in holdout]))
    # df[df["video"] == i]["index"].tolist() returns the indices of the video i
    # then I make a flat list of all the indices of the holdout videos using
    # `itertools.chain`

    vids = list(set(range(len(df))) - set(test_inds))
    # set difference to get the indices of the videos that are not in the test set

    train_inds, val_inds = get_10_folds(vids)

    return train_inds, val_inds, test_inds


def print_and_log(to_print: str, logs: List[io.TextIOWrapper]) -> None:
    '''
    Print and log the message.
    
    Args
    ----
    to_print: str
        Message to be printed and logged.
    logs: list
        List of log files where the message is to be logged.
        
    Returns
    -------
    None
    '''
    if to_print[-1] != "\n":
        to_print += "\n"
    print(to_print, end="")
    for log in logs:
        log.write(to_print)
        log.flush()

    return


def evaluate(preds: np.ndarray, labels: np.ndarray,
             task: str) -> Tuple[float, float, float]:
    '''
    Evaluates the model's predictions based on the task.
    
    Args
    ----
    preds: np.ndarray
        Predictions made by the model.
    labels: np.ndarray
        True labels.
    task: str
        Task of the model. Either "classification" or "regression".
    
    Returns
    -------
    Tuple of floats
        Evaluation metrics based on the task.
        For classification: accuracy, f1, auc (in that order)
        For regression: mse, mae, pearson (in that order)
    '''
    if task == "classification":
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        auc = roc_auc_score(labels, preds)

        return acc, f1, auc

    elif task == "regression":
        mse = np.mean((preds - labels)**2)
        mae = np.mean(np.abs(preds - labels))
        pearson = pearsonr(preds, labels).statistic

        return mse, mae, pearson
    else:
        raise NotImplementedError


# if __name__ == '__main__':

#     data = pd.read_csv("data/all_gait.csv", index_col=None)
#     train, val, test = train_val_test_split_inds(data)
#     print(len(train[0]), len(val[0]), len(test))
