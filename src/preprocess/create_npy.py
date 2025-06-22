"""
This module is uesd to preprocess the V2 dataset, it creates .npy files (present natively) 
from the CSVs using `csv2npy`.

This is the first preprocessing step for the V2 dataset.
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd


# set the seed for reproducibility
random.seed(42)
np.random.seed(42)


def csv2npy(fname: str) -> np.ndarray:
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

    # this dataset has 33 copies of the original data + augmentations
    # we only need the original data (the first x rows)
    actual = df.shape[0] // 33
    assert actual * 33 == df.shape[0], f"Expected a multiple of 33, got {df.shape[0]}"
    df = df.iloc[:actual]

    # convert to numpy array
    arr = df.to_numpy()
    assert arr.shape[1] == 39, f"Expected 39 columns, got {arr.shape[1]}"

    # reshape to make the last dimension 3
    arr = arr.reshape(-1, 13, 3)
    assert np.allclose(arr[:, :, 0], df.iloc[:, 0::3]), "X coordinates do not match"
    assert np.allclose(arr[:, :, 1], df.iloc[:, 1::3]), "Y coordinates do not match"
    assert np.allclose(arr[:, :, 2], df.iloc[:, 2::3]), "Z coordinates do not match"

    return arr


def main(dataset_ver: int = 2):

    # make the npy file, otherwise keypoints are a list of json files
    if dataset_ver == 1:
        for identifier in range(151):
            path = f"data/final_keypoints/{identifier}"
            frames = sorted(os.listdir(path))
            keypoints = []
            for frame in frames:
                with open(os.path.join(path, frame), "r") as f:
                    data = json.load(f)
                    keypoints.append(np.array(data["people"][0]["pose_keypoints_2d"]).reshape(-1, 3))
            keypoints = np.array(keypoints)
            if not os.path.exists(f"data/final_keypoints/{identifier}/kypts.npy"):
                os.makedirs(f"data/final_keypoints/{identifier}", exist_ok=True)
                np.save(f"data/final_keypoints/{identifier}/kypts.npy", keypoints)

    else:
        # create the keypoints directory
        keypoint_path = "data/V2/keypoints/"
        if not os.path.exists(keypoint_path):
            os.makedirs(keypoint_path)

        # assign each video (total 40) a random number
        ids = np.random.permutation(40)

        # create a dict to-be converted to a df and saved as csv
        dict2csv = {"video": [], "idx": [], "label": []}

        # create the keypoints for the ataxic videos
        for i, csv in enumerate(sorted(os.listdir("data/V2/Cerebellar Ataxic Gait/ataxia_features"))):
            print(f"Processing {csv}...")
            arr = csv2npy(f"data/V2/Cerebellar Ataxic Gait/ataxia_features/{csv}")
            np.save(f"{keypoint_path}/{ids[i]}.npy", arr)
            print(f"Assigned ID: {ids[i]}")
            dict2csv["video"].append(csv)
            dict2csv["idx"].append(ids[i])
            dict2csv["label"].append(1)

        # create the keypoints for the normal videos
        for i, csv in enumerate(sorted(os.listdir("data/V2/Normal Gait/normal_features"))):
            print(f"Processing {csv}...")
            arr = csv2npy(f"data/V2/Normal Gait/normal_features/{csv}")
            np.save(f"{keypoint_path}/{ids[i+20]}.npy", arr)
            print(f"Assigned ID: {ids[i+20]}")
            dict2csv["video"].append(csv)
            dict2csv["idx"].append(ids[i+20])
            dict2csv["label"].append(0)

        # save the csv
        df = pd.DataFrame.from_dict(dict2csv)
        df = df.sort_values(by="idx") # shuffles the DataFrame
        df.to_csv("data/V2.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        dataset_ver = int(sys.argv[1])
        main(dataset_ver)
    else:
        print("Usage: python create_npy.py <dataset_version>")
        print("Dataset version 1: Original dataset with 151 videos.")
        print("Dataset version 2: V2 dataset with 40 videos (20 ataxic, 20 normal).")
        sys.exit(1)
