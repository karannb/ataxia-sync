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


def seedAll(seed: int):
    """
    Seeds all the random number generators.

    Args:
        seed (int): The seed to be used.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    return


def pLog(to_print: str, logs: List[io.TextIOWrapper]):
    """
    Print and log the message.

    Args:
        to_print (str): Message to be printed and logged.
        logs (List[io.TextIOWrapper]): List of log files where the message is to be logged.
    """
    # print with new-line
    if to_print[-1] != "\n":
        to_print += "\n"
    for log in logs:
        log.write(to_print)
        log.flush()
    print(to_print, end="")


def getKFolds(train_val_inds, k: int = 10, shuffle: bool = True) -> Tuple[List[List[int]], 
                                                                          List[List[int]]]:
    """
    Splits the data into k folds for cross-validation.


    DO NOT CALL THIS FUNCTION DIRECTLY. USE `getTrainValTest` INSTEAD.


    Args:
        train_val_inds: List of indices to be split into 10 folds.
        k (int, optional): Number of folds. Defaults to 10.
        shuffle (bool, optional): to shuffle the indices or not. Defaults to True.

    Raises:
        NotImplementedError: is train_val_inds is not a List

    Returns:
        Tuple[List[List[int]], List[List[int]]]: train_inds and val_inds (in that order)
    """

    if isinstance(train_val_inds, list):
        train_inds = []
        val_inds = []
        total = len(train_val_inds)

        # Shuffle the indices if required
        if shuffle:
            random.shuffle(train_val_inds)

        # Split the data into 10 folds
        for i in range(1, k + 1):
            # Get the start and end indices for the validation set
            start_idx = int(0.1 * total * (i - 1))
            end_idx = int(0.1 * total * i)
            # Append the training and validation indices to the respective lists
            val_inds.append(train_val_inds[start_idx:end_idx])
            # Concatenate the training indices before and after the validation indices
            train_inds.append(train_val_inds[:start_idx] + train_val_inds[end_idx:])
        return train_inds, val_inds
    else:
        raise NotImplementedError("train_val_inds should be a list")


def getTrainValTest(df: pd.DataFrame, task: str,
                    dataset_ver: int = 1,
                    do_test_split: bool = False, 
                    shuffle: bool = True) -> Tuple[List[List[int]], 
                                                   List[List[int]], 
                                                   List[int]]:
    """
    Splits the data into training, validation, and testing sets.
    Note dataset 2 has a different evaluation scheme, described in Section IV
    https://hisham246.github.io/uploads/iecbes2022khalil.pdf.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        task (str): The task being optimized for (classification/regression).
        dataset_ver (int, optional): The version of the dataset to be used. Defaults to 1.
        do_test_split (bool, optional): Whether to split the data into testing set or not. Defaults to False.
        shuffle (bool, optional): To shuffle the indices or not. Defaults to True.

    Returns:
        Tuple[List[List[int]], List[List[int]], List[int]]: train_inds, val_inds and test_inds (in that order)
        test_inds = [] if do_test_split is False.
    """
    if do_test_split:
        print("We validated our research through 10 fold cross-validation.")
        print("Thus, we do not endorse any results using a separate holdout set for evaluation.")
        if task == "classification":
            # get the indices of the videos that have label 1 and 0
            # so that the holdout set is balanced
            vids_plus = [int(x) for x in os.listdir("data/gait_cycles/") if \
                (df[df["video"] == int(x)]["label"].values[0] == 1)]
            vids_minus = [int(x) for x in os.listdir("data/gait_cycles/") if \
                (df[df["video"] == int(x)]["label"].values[0] == 0)]
            holdout = random.sample(vids_plus, k=5)
            holdout.extend(random.sample(vids_minus, k=5))
        else:
            # for regression, just randomly sample 10 videos
            vids = [int(x) for x in os.listdir("data/gait_cycles/")]
            holdout = random.sample(vids, k=10)

        test_inds = list(itertools.chain(*[df[df["video"] == i]["index"].tolist() for i in holdout]))
        # df[df["video"] == i]["index"].tolist() returns the indices of the video i
        # then make a flat list of all the indices of the holdout videos using
        # `itertools.chain`

    else:
        test_inds = []

    # set difference to get the indices of the videos that are not in the test set
    vids = list(set(range(len(df))) - set(test_inds))

    # get the train and val splits
    if dataset_ver == 1:
        train_inds, val_inds = getKFolds(vids, shuffle)
    else:
        assert test_inds == [], "Datset 2 requires do_test_split to be False."
        # for fold 1 we keep split the ataxic and non-ataxic 
        # videos of each person equally between the training 
        # and validation sets
        mapping = pd.read_csv("data/V2.csv")
        train, val = [], []
        for i in range(1, 21):
            ataxic_id = mapping[mapping["video"] == f"ataxia_features_{i}.csv"]["idx"].values[0]
            normal_id = mapping[mapping["video"] == f"normal_features_{i}.csv"]["idx"].values[0]
            ataxic = df[(df["video"] == ataxic_id) & (df["label"] == 1)]["index"].tolist() # get the indices of the ataxic videos
            normal = df[(df["video"] == normal_id) & (df["label"] == 0)]["index"].tolist() # get the indices of the normal videos
            # make the splits deterministic for reproducibility
            train.extend(ataxic[:len(ataxic)//2] + normal[:len(normal)//2])
            val.extend(ataxic[len(ataxic)//2:] + normal[len(normal)//2:])

        # these are folds 2-5
        train_inds, val_inds = getKFolds(vids, 4, shuffle)
        train_inds.insert(0, train)
        val_inds.insert(0, val)

        # also save for reproducibility 
        # (seedAll is always called at the 
        # start of an experiment, so this 
        # is reproducible)
        with open("data/V2_train_inds.txt", "w") as f:
            for i in range(5):
                f.write("Train " + str(i) + ":\n")
                f.write(str(train_inds[i]) + "\n")
                f.write("Val " + str(i) + ":\n")
                f.write(str(val_inds[i]) + "\n")

    return train_inds, val_inds, test_inds


def evaluate(preds: np.ndarray, labels: np.ndarray,
             task: str) -> Tuple[float, float, float]:
    """
    Evaluates the model's predictions based on the task.

    Args:
        preds (np.ndarray): Predictions made by the model.
        labels (np.ndarray): True labels.
        task (str): Task of the model. Either "classification" or "regression".

    Raises:
        NotImplementedError: If the task passed is not regression or classification.

    Returns:
        Tuple[float, float, float]: Evaluation metrics based on the task.
        For classification: accuracy, f1, auc (in that order)
        For regression: mse, mae, pearson (in that order)
    """
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


def get_mean_and_std(csv_path: str, round_off=True):
    """
    Utility function to analyse the results from a directory,
    following our logging format.
    """

    csv = pd.read_csv(csv_path)

    cols = []
    mus = []
    sigmas = []

    for col in csv.columns:

        cols.append(col)
        mu = csv[col].mean()
        sigma = csv[col].std()
        if round_off:
            mu = round(mu, 4)
            sigma = round(sigma, 4)
        mus.append(mu)
        sigmas.append(sigma)

    new_df = pd.DataFrame({"Name": cols, "Mu": mus, "Sigma": sigmas})

    print(new_df)
