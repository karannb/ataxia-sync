import torch
import numpy as np
from torch.utils.data import Dataset


def splitter(length) -> tuple:
    inds = np.arange(length)
    np.random.shuffle(inds)
    return inds[:119], inds[119:]


class ATAXIA(Dataset):

    def __init__(self, inds=None, data_path="data/"):
        # Load data
        self.data = np.load(f"{data_path}/X_combined.npy")
        self.labels = np.load(f"{data_path}/y_combined.npy")

        # Preprocess data
        self.preprocess()

        # Load the split
        assert inds is not None, "Please provide the split indices"
        self.data = self.data[inds]
        self.labels = self.labels[inds]

        if len(inds) == 30:  # FIXME : Hardcoded
            # print distribution of labels
            print("Distribution of labels in test set")
            print(np.unique(self.labels, return_counts=True))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def preprocess(self):
        # Preprocess data, added M_in as 1 because we have
        # only one instance in each frame
        self.data = self.data[:, :, :19, :, np.newaxis]  # (batch, T, V, channel, M)
        # Picked 19 first keypoints now reassign starting
        # from 9 to make it into a proper 18-keypoint format
        # i.e. make the 9th keypoint the 8th, 10th the 9th and so on
        self.data = self.data[
            :, :, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], :, :
        ]  # NOTE : 8th is not selected
        self.data = np.transpose(
            self.data, (0, 3, 1, 2, 4)
        )  # (batch, channel, T, V, M)
        # make torch tensor
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.int64)


# if __name__ == '__main__':
#     dataset = ATAXIA()
