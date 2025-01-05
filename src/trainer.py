import sys
sys.path.append("./")

import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.MLP import MLP
from src.dataset import ATAXIADataset
from models.atgcn import TruncatedSTGCN, TruncatedResGCN
from src.utils import seedAll, getTrainValTest, evaluate, pLog


class TrainArgs:
    # logging params
    with_tracking: bool = False
    log_dir: str = "results/graph_gait_debug"
    log_every: int = 10

    # training params
    dataset_ver: int = 1
    overlapping: bool = False
    do_test_split: bool = False
    task: str = "classification"
    shuffle: bool = True
    batch_size: int = 256
    epochs: int = 1000
    lr: float = 3e-5
    weight_decay: float = 0.0
    folds: int = 10 # DO NOT SET MANUALLY
    eval_every: int = 10
    save_every: int = 50
    patience: int = 50
    seed: int = 42


class ModelArgs:
    model_type: str = "stgcn"
    layer_num: int = 4
    use_mlp: bool = False
    freeze_encoder: bool = False
    ckpt_path: str = "models/st_gcn.kinetics.pt"


def parse_args() -> Tuple[TrainArgs, ModelArgs]:
    """
    Parse the arguments for the training and model.
    Available arguments are:
        --dataset_ver: int, which dataset version to use (1 or 2).
        --with_tracking: bool, log to wandb or not. Default: False
        --log_every: int, log every log_every epochs. Default: 10
        --log_dir: str, where to log the results. Default: "save"
        --overlapping: bool, use overlapping GAIT cycles. Default: False
        --task: str, the task to be performed. Default: "classification"
        --no_shuffle: bool, shuffle the train-val split or not. Default: False
        --do_test_split: bool, do a test split or not. Default: False
        --batch_size: int, batch size. Default: 256
        --epochs: int, number of epochs to train for. Default: 1000
        --lr: float, learning rate. Default: 3e-5
        --weight_decay: float, weight decay for all parameters. Default: 0.0
        --eval_every: int, evaluate every eval_every epochs. Default: 10
        --save_every: int, save model every save_every epochs. Default: 50
        --patience: int, early stopping patience. Default: 50
        --seed: int, seed the experiment. Default: 42
        --model_type: str, the type of model to be used (stgcn/resgcn). Default: "stgcn"
        --layer_num: int, decides which block of STGCN (Or MLP if set to -2) is to be used. Default: 4
        --freeze_encoder: bool, freeze the encoder (i.e. STGCN) or not. Default: False
        --use_mlp: bool, use a MLP instead of a Conv2d. Default: False
        --ckpt_path: str, path to the checkpoint file. Default: "ckpts/st_gcn.kinetics.pt"

    Returns:
        Tuple[TrainArgs, ModelArgs]: The training and model arguments.
    """

    parser = ArgumentParser()

    # Logging params
    parser.add_argument("--dataset_ver",
                        type=int,
                        required=True,
                        choices=[1, 2],
                        help="Dataset version to use (1 or 2).")
    parser.add_argument("--with_tracking",
                        default=False,
                        action="store_true",
                        help="Wether to track with w&b or not.",
                        )
    parser.add_argument("--log_dir",
                        default="results/ovr_debug_final",
                        type=str,
                        help="Where to log the results. Specify without a '/' or '\\' at the end, e.g. 'save' and not 'save/' "
                        )
    parser.add_argument("--log_every",
                        default=10,
                        type=int,
                        help="Logs to w&b (and terminal) every log_every epochs."
                        )

    # Training params
    parser.add_argument("--overlapping",
                        default=False,
                        action='store_true',
                        help="Use non-overlapping GAIT cycles.")
    parser.add_argument("--do_test_split",
                        default=False,
                        action="store_true",
                        help="Whether to do a test split or not.",
                        )
    parser.add_argument("--task",
                        default="classification",
                        choices=["classification", "regression"],
                        help="The task to be performed. (classification/regression)",
                        )
    parser.add_argument("--no_shuffle",
                        default=False,
                        action="store_true",
                        help="Wether to shuffle the train-val split or not.",
                        )
    parser.add_argument("--batch_size",
                        "-b",
                        type=int,
                        default=64,
                        help="Select Batch Size.")
    parser.add_argument("--epochs",
                        "-e",
                        type=int,
                        default=500,
                        help="Number of epochs to train for.")
    parser.add_argument("--lr",
                        type=float,
                        default=3e-5,
                        help="Select Learning Rate.")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.0,
                        help="Weight Decay for all parameters.")
    parser.add_argument("--eval_every",
                        type=int,
                        default=20,
                        help="Evaluate every eval_every epochs.")
    parser.add_argument("--save_every",
                        type=int,
                        default=100,
                        help="Save model every save_every epochs.")
    parser.add_argument("--patience",
                        type=int,
                        default=20,
                        help="Early stopping patience.")
    parser.add_argument("--seed",
                        type=int,
                        default=40,
                        help="Seeds the experiment.")

    # Model params
    parser.add_argument("--model_type",
                        default="stgcn",
                        choices=["stgcn", "resgcn"],
                        help="The type of model to be used (stgcn/resgcn)."
                        )
    parser.add_argument("--layer_num",
                        type=int,
                        default=4,
                        help="Decides which block of STGCN (Or MLP if set to -2) is to be used."
                        )
    parser.add_argument("--freeze_encoder",
                        default=False,
                        action='store_true',
                        help="Wether or not to freeze the encoder (i.e. STGCN)."
                        )
    parser.add_argument("--use_mlp",
                        default=False,
                        action="store_true",
                        help="Use a MLP instead of a Conv2d."
                        )
    parser.add_argument("--ckpt_path",
                        default="ckpts/st_gcn.kinetics.pt",
                        help="Path to the checkpoint file. Passing 'None' will not load any checkpoint.",
                        )

    args = parser.parse_args()

    train_args = TrainArgs()
    train_args.dataset_ver = args.dataset_ver
    if train_args.dataset_ver == 2:
        print("Training from scratch.")
    train_args.with_tracking = args.with_tracking
    train_args.log_dir = args.log_dir
    train_args.overlapping = args.overlapping
    train_args.do_test_split = args.do_test_split
    train_args.task = args.task
    train_args.shuffle = not args.no_shuffle
    train_args.log_every = args.log_every
    train_args.batch_size = args.batch_size
    train_args.epochs = args.epochs
    train_args.lr = args.lr
    train_args.weight_decay = args.weight_decay
    train_args.eval_every = args.eval_every
    train_args.save_every = args.save_every
    train_args.patience = args.patience
    train_args.seed = args.seed

    seedAll(train_args.seed)

    model_args = ModelArgs()
    model_args.model_type = args.model_type
    model_args.layer_num = args.layer_num
    model_args.use_mlp = args.use_mlp
    model_args.ckpt_path = args.ckpt_path
    model_args.freeze_encoder = args.freeze_encoder

    return train_args, model_args


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader,
             task: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Validates the model on the given data.
    
    Args
    ----
    model: nn.Module
        The model to be validated.
    loader: DataLoader
        The DataLoader object for the validation data.
    task: str
        The task being performed (regression/classification)
    
    Returns
    -------
    preds: np.ndarray
        The predicted labels.
    labels: np.ndarray
        The true labels
    mean_loss: float
        The mean loss on the validation data.
    """

    model.eval()
    preds = []
    labels = []
    loss = []
    for _, (X, y) in enumerate(loader):

        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()

        if task == "classification":
            features = F.softmax(model(X), dim=-1)
            test_loss = F.cross_entropy(features, y)
            preds.append(features.argmax(dim=1).cpu().numpy())
        elif task == "regression":
            features = model(X).squeeze(1)
            test_loss = F.mse_loss(features, y)
            preds.append(features.cpu().numpy())
        loss.append(test_loss.item())
        labels.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return preds, labels, sum(loss) / len(loss)


def train_step(model: nn.Module, loader: DataLoader,
               optimizer: torch.optim.Optimizer, criterion: nn.Module,
               task: str) -> float:
    """
    Trains the model for one epoch.
    
    Args
    ----
    model: nn.Module
        The model to be trained.
    loader: DataLoader
        The DataLoader object for the training data.
    optimizer: torch.optim.Optimizer
        The optimizer to be used.
    criterion: nn.Module
        The loss function to be used.
    task: str
        The task to be performed. (classification/regression)
    """

    model.train()
    losses = []
    for _, (X, y) in enumerate(loader):
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
        if task == "classification":
            features = F.softmax(model(X), dim=-1)
        elif task == "regression":
            features = model(X).squeeze(1)
        loss = criterion(features, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return sum(losses) / len(losses)


def trainer():

    # Parse the arguments
    train_args, model_args = parse_args()

    # Logging params and files
    if train_args.with_tracking:
        import wandb

    if train_args.task == "classification":
        ovr_results = {"Test Accuracy": [], "Test F1": [], "Test AUC": []}
    else:
        ovr_results = {"Test MSE": [], "Test MAE": [], "Test Pearson": []}

    # Create the directory for saving the logs and checkpoints
    ovr_save_pth = f"dataset_{train_args.dataset_ver}_task_{train_args.task}_frozen_encoder_{model_args.freeze_encoder}_shuffle_{train_args.shuffle}_epochs_{train_args.epochs}_seed_{train_args.seed}_lr_{train_args.lr}_bs_{train_args.batch_size}_wd_{train_args.weight_decay}_folds_{train_args.folds}_layer_{model_args.layer_num}_mlp_{model_args.use_mlp}/"

    # In case the directories doesn't exist
    if not os.path.exists(train_args.log_dir):
        os.mkdir(train_args.log_dir)
    if not os.path.exists(f"{train_args.log_dir}/" + ovr_save_pth):
        os.mkdir(f"{train_args.log_dir}/" + ovr_save_pth)

    # Open the log file
    ovr_log = open(f"{train_args.log_dir}/{ovr_save_pth}/ovr.log", "w")

    # Overall dataset file name
    if train_args.overlapping:
        csv_name = "all_gait"
    else:
        csv_name = "non_overlapping_all_gait"

    # Load the data
    data = pd.read_csv("data/" + csv_name + ".csv", index_col=None)

    # Split the data
    train_inds, val_inds, test_inds = getTrainValTest(data, train_args.task, 
                                                      train_args.do_test_split, train_args.shuffle)

    # Save the test indices if testing is enabled
    if train_args.do_test_split:
        with open(f"{train_args.log_dir}/" + ovr_save_pth + "/test_inds.pkl", "wb") as f:
            pickle.dump((test_inds), f)

    # K-fold cross validation
    for fold, train_split, val_split in zip(range(train_args.folds), train_inds, val_inds):

        # Create a directory for the fold
        fold_save_pth = f"fold_{fold}"
        if not os.path.exists(f"{train_args.log_dir}/" + ovr_save_pth + fold_save_pth):
            os.mkdir(f"{train_args.log_dir}/" + ovr_save_pth + fold_save_pth)

        # Open the log file for this fold
        log = open(f"{train_args.log_dir}/{ovr_save_pth}/{fold_save_pth}/training.log", "w")

        loggers = [log, ovr_log]  # used by the utility function

        # Save the indices for later evaluation (if any)
        with open(f"{train_args.log_dir}/" + ovr_save_pth + fold_save_pth + "/inds.pkl", "wb") as f:
            pickle.dump((train_split, val_split), f)

        # Load data
        train_data = ATAXIADataset(train_args.dataset_ver, train_split, train_args.task, 
                                   csv_name=csv_name, 
                                   model=model_args.model_type)
        val_data = ATAXIADataset(train_args.dataset_ver, val_split, train_args.task, 
                                 csv_name=csv_name, 
                                 model=model_args.model_type)

        # create a test dataset if testing is enabled
        if train_args.do_test_split:
            test_data = ATAXIADataset(train_args.dataset_ver, test_inds, train_args.task, 
                                      csv_name=csv_name, 
                                      model=model_args.model_type)

        # print distribution of labels in the test set.
        if train_args.do_test_split:
            to_print = f"Distribution of labels in the test set : {np.unique(test_data.labels, return_counts=True)}\n"
            pLog(to_print, loggers)

        # print distribution of labels in the test set.
        to_print = f"Distribution of labels in the Val set : {np.unique(val_data.labels, return_counts=True)}\n"
        pLog(to_print, loggers)

        # Create the dataloaders
        train_loader = DataLoader(train_data, batch_size=train_args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=train_args.batch_size, shuffle=False)
        if train_args.do_test_split:
            test_loader = DataLoader(test_data, batch_size=train_args.batch_size, shuffle=False)

        # Load the model
        if model_args.layer_num == -2:  # MLP
            model = MLP(task=train_args.task)
        elif model_args.model_type == "stgcn":
            model = TruncatedSTGCN(model_args.layer_num, model_args.use_mlp,
                                   train_args.task, model_args.freeze_encoder)
            if model_args.ckpt_path != 'None':
                state_dict = torch.load(model_args.ckpt_path)
                model.load_state_dict(state_dict, strict=False)  # strict=False because we are loading a subset of the model
        elif model_args.model_type == "resgcn":
            assert train_args.dataset_ver == 1, "Gait Graph experiments are only supported for dataset V1."
            model = TruncatedResGCN(model_args.layer_num, train_args.task, 
                                    model_args.freeze_encoder)
            if model_args.ckpt_path != 'None':
                state_dict = torch.load(model_args.ckpt_path)
                model.load_state_dict(state_dict, strict=False) # same, our model has a prediction head

        # print the number of trainable parameters
        to_print = f"Number of trainable parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.3f}M\n"
        pLog(to_print, loggers)

        # Move the model to GPU if available
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                print("This code is not optimized to be used with multiple GPUs. Will use only one GPU.")
            model = model.to("cuda:0")

        # Define the optimizer
        if train_args.task == "classification":
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=train_args.lr,
                                     weight_decay=train_args.weight_decay)

        # W&B
        if train_args.with_tracking:
            wandb.init(project="ataxia",
                       config={**vars(train_args), **vars(model_args)},
                       name=str(fold),
                       group=ovr_save_pth)

        # Used for checkpointing and early stopping
        best_model = None
        patience = train_args.patience
        best_acc = 0.0
        best_mae = float('inf')
        acc, f1, auc, mse, mae, pearson = None, None, None, None, None, None

        # Training loop
        for epoch in range(train_args.epochs):

            # One epoch of training
            mean_loss = train_step(model, train_loader, optimizer, criterion, train_args.task)

            # Logging
            if train_args.with_tracking:
                wandb.log({"Train Loss": mean_loss}, step=epoch)

            # Print and log the loss every log_every epochs
            if epoch % train_args.log_every == 0:
                to_print = f"Fold : {fold}, Epoch : {epoch}, Loss : {mean_loss}\n"
                pLog(to_print, loggers)

            # Save the model every save_every epochs
            if epoch % train_args.save_every == 0:
                torch.save(model.state_dict(), f"{train_args.log_dir}/{ovr_save_pth}/{fold_save_pth}/model_{epoch}.pth")

            # Evaluate the model every eval_every epochs
            if epoch % train_args.eval_every == 0:
                # Validation
                preds, labels, val_loss = validate(model, val_loader, train_args.task)

                if train_args.task == "classification":
                    acc, f1, auc = evaluate(preds, labels, train_args.task)
                    to_print = f"Fold : {fold}, Epoch : {epoch}, Validation Accuracy : {acc}, Validation F1 : {f1}, Validation AUC : {auc}\n"
                else:
                    mse, mae, pearson = evaluate(preds, labels, train_args.task)
                    to_print = f"Fold : {fold}, Epoch : {epoch}, Validation MSE : {mse}, Validation MAE : {mae}, Validation Pearson : {pearson}\n"

                # Print and log the validation results for that evaluation
                pLog(to_print, loggers)

                if train_args.with_tracking:
                    if train_args.task == "classification":
                        wandb.log({
                            "Validation Accuracy": acc,
                            "Validation F1": f1,
                            "Validation AUC": auc,
                            "Validation Loss": val_loss,
                            },
                            step=epoch,
                        )
                    else:
                        wandb.log({
                            "Validation MSE": mse,
                            "Validation MAE": mae,
                            "Validation Pearson": pearson,
                            "Validation Loss": val_loss,
                            },
                            step=epoch,
                        )

                # Save the best model and check for early stopping
                if (acc != None and acc > best_acc) or (mae != None and mae < best_mae):
                    patience = train_args.patience
                    best_acc = acc
                    best_mae = mae
                    if train_args.task == "classification":
                        best_model = {"model": model.state_dict(), "epoch": epoch, "f1": f1, "acc": acc, "auc": auc}
                        to_print = f"Best model saved at epoch {epoch} with Accuracy of {acc}\n"
                    else:
                        best_model = {"model": model.state_dict(), "epoch": epoch, "mae": mae, "mse": mse, "pearson": pearson}
                        to_print = f"Best model saved at epoch {epoch} with MAE of {mae}\n"
                    torch.save(best_model, f"{train_args.log_dir}/{ovr_save_pth}/{fold_save_pth}/best_model.pth")
                    pLog(to_print, loggers)
                else:
                    patience -= 1
                    if patience == 0:
                        to_print = "EXITING THROUGH AN EARLY STOP.\n"
                        pLog(to_print, loggers)
                        break

        # Final evaluation on the best model
        state_dict = torch.load(f"{train_args.log_dir}/{ovr_save_pth}/{fold_save_pth}/best_model.pth")["model"]
        model.load_state_dict(state_dict)

        # In case of testing, use the test_loader, else use the val_loader
        if not train_args.do_test_split:
            test_loader = val_loader

        preds, labels, _ = validate(model, test_loader, train_args.task)
        if train_args.task == "classification":
            acc, f1, auc = evaluate(preds, labels, train_args.task)
            to_print = f"Fold : {fold}, Final Evaluation\nTest Accuracy : {acc}, Test F1 : {f1}, Test AUC : {auc}\n"
        else:
            mse, mae, pearson = evaluate(preds, labels, train_args.task)
            to_print = f"Fold : {fold}, Final Evaluation\nTest MSE : {mse}, Test MAE : {mae}, Test Pearson : {pearson}\n"
        pLog(to_print, loggers)

        # close the log file for this fold
        log.close()

        # Terminate wandb run
        if train_args.with_tracking:
            wandb.finish()

        # Save the results for this fold
        if train_args.task == "classification":
            ovr_results["Test Accuracy"].append(acc)
            ovr_results["Test F1"].append(f1)
            ovr_results["Test AUC"].append(auc)
        else:
            ovr_results["Test MSE"].append(mse)
            ovr_results["Test MAE"].append(mae)
            ovr_results["Test Pearson"].append(pearson)

    # Create a dataframe and save the results in a csv
    results = pd.DataFrame(ovr_results)
    results.to_csv(f"{train_args.log_dir}/{ovr_save_pth}/results.csv")

    # Average the results and print them
    if train_args.task == "classification":
        mean_acc = results["Test Accuracy"].mean()
        std_acc = results["Test Accuracy"].std()
        mean_f1 = results["Test F1"].mean()
        std_f1 = results["Test F1"].std()
        mean_auc = results["Test AUC"].mean()
        std_auc = results["Test AUC"].std()
        to_print = f"Test Accuracy : {mean_acc} +/- {std_acc}, Test F1 : {mean_f1} +/- {std_f1}, Test AUC : {mean_auc} +/- {std_auc}\n"
    else:
        mean_mse = results["Test MSE"].mean()
        std_mse = results["Test MSE"].std()
        mean_mae = results["Test MAE"].mean()
        std_mae = results["Test MAE"].std()
        mean_pearson = results["Test Pearson"].mean()
        std_pearson = results["Test Pearson"].std()
        to_print = f"Test MSE : {mean_mse} +/- {std_mse}, Test MAE : {mean_mae} +/- {std_mae}, Test Pearson : {mean_pearson} +/- {std_pearson}\n"
    print(to_print, end="")
    ovr_log.write(to_print)
    ovr_log.close()


if __name__ == "__main__":
    trainer()
