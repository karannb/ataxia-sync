"""
This module is uesd to preprocess the V2 dataset, it creates .npy files from the CSVs
using `csv2npy`.
"""

import os
import random
import numpy as np
import pandas as pd


# set the seed for reproducibility
random.seed(42)
np.random.seed(42)


def csv2npy(fname: str):
    """
    Converts the V2 CSV files to npy arrays by aggregating the X, Y, Z coordinates
    and adding a center coordinate.

    Args:
        fname (str): CSV file name

    Returns:
        np.ndarray: npy array
    """

    assert fname.endswith(".csv"), f"Expected a CSV file, got {fname.split('.')[-1]}"

    # open the CSV file
    df = pd.read_csv(fname, index_col=False)

    # drop useless columns
    cols = df.columns
    drop_cols = cols[-11:] # the last 11 columns are not keypoints, but metadata
    df = df.drop(columns=drop_cols)

    # add a center coordinate
    df["XCenter"] = (df["XRightShoulder"] + df["XLeftShoulder"]) / 2
    df["YCenter"] = (df["YRightShoulder"] + df["YLeftShoulder"]) / 2
    df["ZCenter"] = (df["ZRightShoulder"] + df["ZLeftShoulder"]) / 2

    # convert to numpy array
    arr = df.to_numpy()
    assert arr.shape[1] == 39, f"Expected 36 columns, got {arr.shape[1]}"

    # reshape to make the last dimension 3
    arr = arr.reshape(-1, 13, 3)
    assert np.allclose(arr[:, :, 0], df.iloc[:, 0::3]), "X coordinates do not match"
    assert np.allclose(arr[:, :, 1], df.iloc[:, 1::3]), "Y coordinates do not match"
    assert np.allclose(arr[:, :, 2], df.iloc[:, 2::3]), "Z coordinates do not match"

    return arr


def main():

    # create the keypoints directory
    keypoint_path = "data/V2/keypoints/"
    if not os.path.exists(keypoint_path):
        os.makedirs(keypoint_path)

    # assign each video (total 40) a random number
    ids = np.random.permutation(40)

    # create the keypoints for the ataxic videos
    for i, csv in enumerate(sorted(os.listdir("data/V2/Cerebellar Ataxic Gait/ataxia_features"))):
        print(f"Processing {csv}...")
        arr = csv2npy(f"data/V2/Cerebellar Ataxic Gait/ataxia_features/{csv}")
        np.save(f"{keypoint_path}/{ids[i]}.npy", arr)
        print(f"Assigned ID: {ids[i]}")

    # create the keypoints for the normal videos
    for i, csv in enumerate(sorted(os.listdir("data/V2/Normal Gait/normal_features"))):
        print(f"Processing {csv}...")
        arr = csv2npy(f"data/V2/Normal Gait/normal_features/{csv}")
        np.save(f"{keypoint_path}/{ids[i+20]}.npy", arr)
        print(f"Assigned ID: {ids[i+20]}")


if __name__ == "__main__":
    main()