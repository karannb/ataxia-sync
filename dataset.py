import os
import torch
import numpy as np
from torch.utils.data import Dataset


def splitter(length) -> tuple:
    inds = np.arange(length)
    np.random.shuffle(inds)
    train2test = int(length * 0.8)
    return inds[:train2test], inds[train2test:]


class ATAXIA(Dataset):

    def __init__(self, inds=None, data_path="data"):
        
        # Get the split
        self.data = []
        self.labels = []
        for ind in inds:
            label = np.load(f"{data_path}/final_keypoints/{ind}/label.npy")
            for gait in os.listdir(f"{data_path}/gait_cycles/{ind}"):
                cycle = np.load(f"{data_path}/gait_cycles/{ind}/{gait}")
                data = cycle.copy()
                while data.shape[0] < 75:
                    data = np.concatenate([data, cycle], axis=0)
                data = data[:75]
                self.data.append(data)
                self.labels.append(label)
                
        # Load data
        self.data = np.array(self.data) # np.load(f"{data_path}/X_combined.npy")
        self.labels = np.array(self.labels) # np.load(f"{data_path}/y_combined.npy")

        # Preprocess data
        self.preprocess()

        # Load the split
        # assert inds is not None, "Please provide the split indices"
        # self.data = self.data[inds]
        # self.labels = self.labels[inds]

        # print distribution of labels
        print("Distribution of labels in the set :")
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
#     data = pd.read_csv("data/overall.csv", index_col=None)
#     train, test = splitter(len(data))
#     train = data["video_num"].iloc[train]
#     test = data["video_num"].iloc[test]
#     dataset = ATAXIA(train)
#     dataset2 = ATAXIA(test)