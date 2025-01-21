"""
This module contains the dataset wrapper for the datasets we use, both V1 and V2.
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from typing import Tuple, List


class ATAXIADataset(Dataset):

    def __init__(self,
                 dataset_ver,
                 inds: List,
                 task="classification",
                 data_path="data",
                 csv_name="all_gait",
                 model="stgcn"):
        """
        Args:
            dataset_ver (int): The version of the dataset to be used.
            inds (List): The indices of the data to be used.
            task (str, optional): The task to be performed. (classification or regression) Defaults to "classification".
            data_path (str, optional): The path to the data directory. Defaults to "data".
            csv_name (str, optional): The name of the csv file containing the data. Defaults to "all_gait".
            model (str, optional): The model to be used (one of stgcn or resgcn). Defaults to "stgcn".
            This is required to create proper graph structure for the model.

        Raises:
            NotImplementedError: in case the task is passed incorrectly, i.e., something other than 
            classification / regression.
        """

        # Read df
        df = pd.read_csv(f"{data_path}/{csv_name}.csv")
        self.task = task
        self.dataset_ver = dataset_ver
        self.model = model

        # gait path
        gait_path = "non_overlapping_gait_cycles" if "non_overlapping" in csv_name else "gait_cycles"

        # Load the split
        assert inds is not None, "Please provide the split indices"

        # Get the split data from the dataframe
        self.data = []
        self.labels = []
        for ind in inds:
            record = df.iloc[ind]
            if task == "classification":
                label = record["label"]  # for classification task, the label is already in the csv
            elif task == "regression":
                assert dataset_ver != 2, "Regression task is not supported for dataset version 2."
                label = record["score"]
                if label >= 3:  # for the regression task, we clip the score to 3 as in Section 4.1
                    label = 3
            else:
                raise NotImplementedError

            # Load the gait cycle
            cycle = np.load(f"{data_path}/{gait_path}/{record['video']}/{record['gait']}")
            data = cycle.copy()
            max_length = 75 if dataset_ver == 1 else 120
            while data.shape[0] < max_length:
                # repeat the gait cycle to make it mac_length frames
                data = np.concatenate([data, cycle], axis=0)
            data = data[:max_length]  # clip the gait cycle to mac_length frames
            self.data.append(data)
            self.labels.append(label)

        # Keep the data and labels as numpy arrays
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

        # Preprocess data
        self.preprocess()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.labels[index]

    def preprocess(self):
        """
        Preprocess data in the format STGCN expects input.
        Out shape - (B, T, V, channel, M)

        Raises:
            NotImplementedError: in case the task is passed incorrectly, 
            redundant check.
        """
        if self.dataset_ver == 1:
            # add M_in as 1 because we have
            # only one instance in each frame
            # M_in is required for the STGCN model
            self.data = self.data[:, :, :19, :, np.newaxis]
            # (batch, T, V, channel, M)

            # Picked 19 first keypoints now reassign starting
            # from 9 to make it into a proper 18-keypoint format
            # i.e. make the 9th keypoint the 8th, 10th the 9th and so on
            self.data = self.data[:, :, [
                0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
            ], :, :]  # NOTE : 8th is not selected

            if self.model == "stgcn":
                self.data = np.transpose(self.data, (0, 3, 1, 2, 4))
                # (batch, channel, T, V, M)
            elif self.model == "resgcn":
                # this model again has one less keypoint + temporal sequence length of 60
                self.data = self.data[:, :60] # clip to 60 frames
                self.data = self.data[:, :, [
                    0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10
                ], :, :]
                # now this expects a different format, (N, I, C, T, V)
                # but similar to STGCN, I = M = 1
                self.data = np.transpose(self.data, (0, 4, 3, 1, 2))
        else:
            # same logic as above
            self.data = self.data[:, :, :, :, np.newaxis]
            # (batch, T, V, channel, M)

            # however, note we don't shuffle anything to match the ordering
            # this is because, 
            # 1. we don't have all keypoints required present in the second dataset
            # 2. because of this, we have to re-initialize the starting layers of 
            # the model, now, it is impossible to tell how the left shoulder affected the
            # 41st channel in the second layer and also match it here, so we don't
            # reorder (pseudo reorder) to match the model's trained graph structure

            if self.model == "stgcn":
                self.data = np.transpose(self.data, (0, 3, 1, 2, 4))
                # (batch, channel, T, V, M)
            elif self.model == "resgcn":
                self.data = self.data[:, :60] # clip to 60 frames
                # now this expects a different format, (N, I, C, T, V)
                # but similar to STGCN, I = M = 1
                self.data = np.transpose(self.data, (0, 4, 3, 1, 2))

        # make torch tensor
        self.data = torch.tensor(self.data, dtype=torch.float32)
        if self.task == "classification":
            self.labels = torch.tensor(self.labels, dtype=torch.int64)
        elif self.task == "regression":
            self.labels = torch.tensor(self.labels, dtype=torch.float32)
        else:
            raise NotImplementedError
