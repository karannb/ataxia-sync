import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from typing import Tuple


class ATAXIADataset(Dataset):

    def __init__(self,
                 inds=None,
                 task="classification",
                 data_path="data",
                 csv_name="all_gait",
                 model="stgcn"):
        """
        Args:
            inds (_type_, optional): The indices of the data to be used. Defaults to None.
            task (str, optional): The task to be performed. (classification or regression) Defaults to "classification".
            data_path (str, optional): The path to the data directory. Defaults to "data".
            csv_name (str, optional): The name of the csv file containing the data. Defaults to "all_gait".
            model (str, optional): The model to be used (one of stgcn or resgcn). Defaults to "stgcn".

        Raises:
            NotImplementedError: in case the task is passed incorrectly, i.e., something other than 
            classification / regression.
        """

        # Read df
        df = pd.read_csv(f"{data_path}/{csv_name}.csv")
        self.task = task

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
                label = record["score"]
                if label >= 3:  # for the regression task, we clip the score to 3 as in Section 4.1
                    label = 3
            else:
                raise NotImplementedError

            # Load the gait cycle
            cycle = np.load(f"{data_path}/{gait_path}/{record['video']}/{record['gait']}")
            data = cycle.copy()
            while data.shape[0] < 75:
                # repeat the gait cycle to make it 75 frames
                data = np.concatenate([data, cycle], axis=0)
            data = data[:75]  # clip the gait cycle to 75 frames
            self.data.append(data)
            self.labels.append(label)

        # Keep the data and labels as numpy arrays
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

        # Preprocess data
        self.preprocess(model)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.labels[index]

    def preprocess(self, model):
        """
        Preprocess data in the format STGCN expects input.
        Out shape - (B, T, V, channel, M)

        Raises:
            NotImplementedError: in case the task is passed incorrectly, 
            redundant check.
        """

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

        if model == "stgcn":
            self.data = np.transpose(self.data, (0, 3, 1, 2, 4))
            # (batch, channel, T, V, M)
        elif model == "resgcn":
            # this model again has one less keypoint + temporal sequence length of 60
            self.data = self.data[:, :60] # clip to 60 frames
            self.data = self.data[:, :, [
                0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10
            ], :, :]
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

        return
