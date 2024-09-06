'''
This file holds the code for plotting and analyzing the results of the experiments.
'''
import sys
sys.path.append("./")

import re
import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm

from typing import Tuple, List
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
from src.dataset import ATAXIADataset
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import ConfusionMatrixDisplay
from models.atgcn import TruncatedSTGCN, TruncatedResGCN

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
    checkpoint = torch.load(ckpt, map_location='cpu')
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
    plt.scatter(zero_labels, range(len(zero_labels)), c=label2color[0], label='SARA Score 0')
    plt.scatter(one_labels, range(len(one_labels)), c=label2color[1], label='SARA Score 1')
    plt.scatter(two_labels, range(len(two_labels)), c=label2color[2], label='SARA Score 2')
    plt.scatter(three_labels, range(len(three_labels)), c=label2color[3], label='SARA Score 3')
    plt.xlabel("Predicted SARA severity score")
    plt.ylabel("Frequency")
    # Add a vertical line at every 0.5 for reference
    for i in range(1, 4):
        plt.axvline(x=i - 0.5, color="k", linestyle="--")
    # Create a legend and place it inside the plot at the top-right corner
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=label2color[i], markersize=10, label=f'SARA Score {i}') for i in label2color]
    plt.legend(handles=legend_elements, title="Ground Truth", loc="upper right")
    # Adjust the layout
    plt.tight_layout()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("data/scatter.png")
    plt.close('all')

    # plot confusion matrix
    preds = np.digitize(preds, [0.5, 1.5, 2.5])
    ConfusionMatrixDisplay.from_predictions(targets, preds)
    plt.savefig("data/confusion_matrix.png")
    plt.close('all')

    return


def getMetricStdandMean(base_results_path: str, layer: int = 6, 
                        task: str = "classification") -> Tuple[float, float]:
    """
    Function to get the mean and standard deviation of the metrics from the base_ckpt_path.

    Args:
        base_results_path (str): The base path to the checkpoints.
        layer (int, optional): The layer to get the metrics for. Defaults to 6.
        task (str, optional): The task to get the metrics for. Defaults to "classification".

    Returns:
        None, just prints the mean and standard deviation of the metrics.
    """
    if task == "classification":
        metrics = {"Test Accuracy": [], "Test F1": [], "Test AUC": []}
    else:
        metrics = {"Test MAE": [], "Test MSE": [], "Test Pearson": []}
    # add results/ if not in base_results_path
    if "results/" not in base_results_path:
        base_results_path = f"results/{base_results_path}"

    # now iterate over each directory in the base_results_path
    # and get the 10-fold metrics for that run
    for folder in os.listdir(base_results_path):
        if (f"layer_{layer}" not in folder) or (task not in folder):
            continue
        csv = pd.read_csv(f"{base_results_path}/{folder}/results.csv")
        for metric in metrics:
            metrics[metric].append(csv[metric].values)
        
    for metric in metrics:
        metrics[metric] = np.array(metrics[metric]).flatten()
        mean = np.mean(metrics[metric])
        std = np.std(metrics[metric])
        if task == "classification":
            print(f"{metric}: {mean*100:.2f} +/- {std*100:.2f}")
        else:
            print(f"{metric}: {mean:.4f} +/- {std:.4f}")

    return


def getParams(model_path: str):
    """
    Function to get the number of parameters in the model.
    Head + Non-head.

    Args:
        model_path (str): The path to the model.

    Returns:
        None, just prints the number of parameters in the model.
    """
    # get layer number
    match = re.search(r'layer_(\-?\d+)', model_path)
    if match:
        layer_num = match.group(1)
        print(f"LAYER: {layer_num}")
    else:
        print("Layer number not found.")
        return
    # get task
    match = re.search(r'task_([^_]+)', model_path)
    if match:
        task_name = match.group(1)
    else:
        print("Task not found.")
        return
    if not os.path.exists(model_path):
        print("Model path not found.")
        return
    state_dict = torch.load(model_path, map_location='cpu')['model']
    if "gaitgraph" in model_path:
        model = TruncatedResGCN(layer=int(layer_num), task=task_name)
    else:
        model = TruncatedSTGCN(layer=int(layer_num), task=task_name)
    head = 0
    non_head = 0
    for name, param in model.named_parameters():
        # don't count non-trianable parameters
        # because in my experiments, non-training parameters
        # are never used, they are the layers of the model
        # beyond the one we want to get the representation of
        if not param.requires_grad:
            continue
        if "head" in name:
            head += param.numel()
        else:
            non_head += param.numel()
    print(f"Head: {head}")
    print(f"Non-Head: {non_head/1e6:.3f}M")
    print(f"Total: {(head + non_head)/1e6:.3f}M")
    return


if __name__ == '__main__':
    pass
    # model = TruncatedSTGCN(layer=6, task="regression")
    # ckpt = "results/fivesixseven_cls/task_regression_frozen_encoder_False_deepnet_False_shuffle_True_epochs_500_seed_42_lr_3e-05_bs_64_wd_0.0_folds_10_layer_6_mlp_False/fold_4/best_model.pth"
    # plotResults(model, ckpt)
    # print("STGCN-5 on Regression:")
    # getMetricStdandMean("fivesixseven_cls", layer=5, task="regression")
    # print("*"*88)
    # print("STGCN-6 on Regression:")
    # getMetricStdandMean("fivesixseven_cls", layer=6, task="regression")
    # print("*"*88)
    # print("STGCN-7 on Regression:")
    # getMetricStdandMean("fivesixseven_cls", layer=7, task="regression")
    # print("*"*88)
    # print("STGCN-5 on Classification:")
    # getMetricStdandMean("fivesixseven_reg", layer=5, task="classification")
    # print("*"*88)
    # print("STGCN-6 on Classification:")
    # getMetricStdandMean("fivesixseven_reg", layer=6, task="classification")
    # print("*"*88)
    # print("STGCN-7 on Classification:")
    # getMetricStdandMean("fivesixseven_reg", layer=7, task="classification")
    # print("STGCN Params from ablation on regression:")
    # for i in range(-1, 10):
    #     getParams(f"results/ablation/task_regression_frozen_encoder_False_deepnet_False_shuffle_True_epochs_500_seed_50_lr_3e-05_bs_64_wd_0.0_folds_10_layer_{i}_mlp_False/fold_0/best_model.pth")
    # print("*"*88)
    # print("STGCN Params on classification:")
    # for i in range(5, 8):
    #     getParams(f"results/fivesixseven_reg/task_classification_frozen_encoder_False_deepnet_False_shuffle_True_epochs_500_seed_40_lr_3e-05_bs_64_wd_0.0_folds_10_layer_{i}_mlp_False/fold_0/best_model.pth")
    # print("*"*88)
    # print("GaitGraph on Classification:")
    # getMetricStdandMean("results/gaitgraph", task="classification", layer=-1)
    # print("*"*88)
    # print("GaitGraph on Regression:")
    # getMetricStdandMean("results/gaitgraph", task="regression", layer=-1)
    # print("*"*88)
    # print("GaitGraph Classification Params:")
    # getParams("results/gaitgraph/task_classification_frozen_encoder_False_shuffle_True_epochs_500_seed_40_lr_3e-05_bs_64_wd_0.0_folds_10_layer_-1_mlp_False/fold_0/best_model.pth")
    # print("*"*88)
    # print("GaitGraph Regression Params:")
    # getParams("results/gaitgraph/task_regression_frozen_encoder_False_shuffle_True_epochs_500_seed_40_lr_3e-05_bs_64_wd_0.0_folds_10_layer_-1_mlp_False/fold_0/best_model.pth")
