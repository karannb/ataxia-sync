'''
This file holds the code for plotting the results of the experiments.
'''
import sys
sys.path.append("./")

import torch
import numpy as np
from torch import nn
from tqdm import tqdm

from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
from src.dataset import ATAXIADataset
from models.atgcn import TruncatedSTGCN
from sklearn.metrics import ConfusionMatrixDisplay
from typing import Tuple, List

@torch.no_grad()
def getPredsAndTargets(model, loader, device) -> Tuple[List, List]:
    """
    Simple function to get the predictions and targets from the model and loader.

    Args:
        model (nn.Module): The model to get the predictions from.
        loader (Dataset / DataLoader): The loader to get the predictions from.
        device ("cuda" or "cpu"): The device to run the model on.

    Returns:
        Tuple[List, List]: The predictions and targets from the model and loader (in that order).
    """
    model.eval()
    model.to(device)
    preds = []
    targets = []
    for data in tqdm(loader):
        inputs, labels = data
        inputs = inputs.to(device).unsqueeze(0)
        labels = labels.to(device).unsqueeze(0)
        outputs = model(inputs)
        preds.append(outputs.cpu().detach())
        targets.append(labels.cpu().detach())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return preds, targets


def plotResults(model: nn.Module, ckpt: str):
    """
    Plots a histogram and a confusion matrix of the model's predictions (regression task).

    Args:
        model (nn.Module): The model to plot the results for.
        ckpt (str): The path to the checkpoint of the model.
    """

    # Load the checkpoint
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model'])

    # get data (377 is dataset size, so we use all the data)
    data = ATAXIADataset(inds=torch.arange(377).tolist(), task="regression", 
                         csv_name="non_overlapping_all_gait", 
                         model="stgcn")

    # get predictions and targets
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds, targets = getPredsAndTargets(model, data, device)

    # plot histogram
    label2color = {0: "r", 1: "g", 2: "b", 3: "y"}
    # Separate the predictions based on the labels
    zero_labels = [preds[i] for i in range(len(targets)) if targets[i] == 0]
    one_labels = [preds[i] for i in range(len(targets)) if targets[i] == 1]
    two_labels = [preds[i] for i in range(len(targets)) if targets[i] == 2]
    three_labels = [preds[i] for i in range(len(targets)) if targets[i] == 3]
    # Plot each class separately with adjusted y-values
    plt.scatter(zero_labels, range(len(zero_labels)), c=label2color[0], label='Label 0')
    plt.scatter(one_labels, range(len(one_labels)), c=label2color[1], label='Label 1')
    plt.scatter(two_labels, range(len(two_labels)), c=label2color[2], label='Label 2')
    plt.scatter(three_labels, range(len(three_labels)), c=label2color[3], label='Label 3')
    plt.xlabel("Predictions")
    plt.ylabel("Number of Points in Class")
    # Add a vertical line at every 0.5 for reference
    for i in range(1, 4):
        plt.axvline(x=i - 0.5, color="k", linestyle="--")
    # Create a legend and place it inside the plot at the top-right corner
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=label2color[i], markersize=10, label=f'Label {i}') for i in label2color]
    plt.legend(handles=legend_elements, title="Labels", loc="upper right")
    # Adjust the layout
    plt.tight_layout()
    plt.savefig("data/scatter.png")
    plt.close('all')

    # plot confusion matrix
    preds = np.digitize(preds, [0.5, 1.5, 2.5])
    ConfusionMatrixDisplay.from_predictions(targets, preds)
    plt.savefig("data/confusion_matrix.png")
    plt.close('all')

    return


if __name__ == '__main__':
    model = TruncatedSTGCN(layer=6, task="regression")
    ckpt = "results/fivesixseven_cls/task_regression_frozen_encoder_False_deepnet_False_shuffle_True_epochs_500_seed_42_lr_3e-05_bs_64_wd_0.0_folds_10_layer_6_mlp_False/fold_4/best_model.pth"
    plotResults(model, ckpt)