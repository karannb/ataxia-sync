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

from utils import *
from model.MLP import MLP
from dataset import ATAXIA
from model.st_gcn import TruncatedModel


class TrainArgs:
    with_tracking: bool = False
    task: str = "classification"
    run_name: str
    log_every: int = 10
    batch_size: int = 256
    epochs: int = 1000
    lr: float = 3e-5
    weight_decay: float = 0.0
    folds: int = 10
    eval_every: int = 10
    save_every: int = 50
    patience: int = 50
    seed: int = 42


class ModelArgs:
    layer_num: int = 4
    use_mlp: bool = False
    ensemble: bool = False
    ckpt_path: str = "models/st_gcn.kinetics.pt"


def parse_args() -> Tuple[TrainArgs, ModelArgs]:
    '''
    Args available -
    --with_tracking: bool = False
    --task: str = "classification"
    --log_every: int = 10
    --batch_size: int = 256
    --epochs: int = 1000
    --lr: float = 3e-5
    --weight_decay: float = 0.0
    --folds: int = 10
    --eval_every: int = 10
    --save_every: int = 50
    --patience: int = 50
    --seed: int = 42
    --layer_num: int = 4
    --use_mlp: bool = False
    --ckpt_path: str = "models/st_gcn.kinetics.pt"
    '''

    parser = ArgumentParser()

    # Logging params
    parser.add_argument(
        "--with_tracking",
        default=False,
        action="store_true",
        help="Wether to track with w&b or not.",
    )
    parser.add_argument(
        "--log_every",
        default=10,
        type=int,
        help="Logs to w&b (and terminal) every log_every epochs.",
    )

    # Training params
    parser.add_argument(
        "--task",
        default="classification",
        choices=["classification", "regression"],
        help="The task to be performed. (classification/regression)",
    )
    parser.add_argument("-b",
                        "--batch_size",
                        type=int,
                        default=256,
                        help="Select Batch Size.")
    parser.add_argument("-e",
                        "--epochs",
                        type=int,
                        default=1000,
                        help="Number of epochs to train for.")
    parser.add_argument("--lr",
                        type=float,
                        default=3e-5,
                        help="Select Learning Rate.")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.0,
                        help="Weight Decay for all parameters.")
    parser.add_argument("--folds",
                        type=int,
                        default=10,
                        help="Number of folds for cross validation.")
    parser.add_argument("--eval_every",
                        type=int,
                        default=10,
                        help="Evaluate every eval_every epochs.")
    parser.add_argument("--save_every",
                        type=int,
                        default=50,
                        help="Save model every save_every epochs.")
    parser.add_argument("--patience",
                        type=int,
                        default=50,
                        help="Early stopping patience.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Seeds the experiment.")

    # Model params
    parser.add_argument(
        "--layer_num",
        type=int,
        default=4,
        help="Decides which block of STGCN (Or MLP if set to -2) is to be used."
    )
    parser.add_argument(
        "--use_mlp",
        default=False,
        action="store_true",
        help="Use a MLP instead of a Conv2d.",
    )
    parser.add_argument(
        "--ckpt_path",
        default="ckpts/st_gcn.kinetics.pt",
        help="Path to the checkpoint file.",
    )

    args = parser.parse_args()

    train_args = TrainArgs()
    train_args.with_tracking = args.with_tracking
    train_args.task = args.task
    train_args.log_every = args.log_every
    train_args.batch_size = args.batch_size
    train_args.epochs = args.epochs
    train_args.lr = args.lr
    train_args.weight_decay = args.weight_decay
    train_args.folds = args.folds
    train_args.eval_every = args.eval_every
    train_args.save_every = args.save_every
    train_args.patience = args.patience
    train_args.seed = args.seed

    seed_all(train_args.seed)

    model_args = ModelArgs()
    model_args.layer_num = args.layer_num
    model_args.use_mlp = args.use_mlp
    model_args.ckpt_path = args.ckpt_path

    return train_args, model_args


@torch.no_grad()
def validate(model: nn.Module,
             loader: DataLoader,
             task: str) -> Tuple[np.ndarray, np.ndarray, float]:
    '''
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
    '''

    model.eval()
    preds = []
    labels = []
    loss = []
    for i, (X, y) in enumerate(loader):

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
    '''
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
    '''

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


def main():

    # Parse the arguments
    train_args, model_args = parse_args()

    # Logging params and files
    if train_args.with_tracking:
        import wandb
    if train_args.task == "classification":
        ovr_results = {
            "Test Accuracy": [],
            "Test F1": [],
            "Test AUC": [],
            "Test Positives": [],
            "Test Negatives": [],
            "Pred Positives": [],
            "Pred Negatives": []
        }
        # Last 4 can be used for confusion matrix
    else:
        ovr_results = {"Test MSE": [], "Test MAE": [], "Test Pearson": []}

    ovr_save_pth = f"task_{train_args.task}_epochs_{train_args.epochs}_seed_{train_args.seed}_lr_{train_args.lr}_bs_{train_args.batch_size}_wd_{train_args.weight_decay}_folds_{train_args.folds}_layer_{model_args.layer_num}_mlp_{model_args.use_mlp}_ensemble_{model_args.ensemble}/"
    if not os.path.exists("save/" + ovr_save_pth):
        os.mkdir("save/" + ovr_save_pth)

    ovr_log = open(f"save/{ovr_save_pth}/ovr.log", "w")

    # Overall dataset
    data = pd.read_csv("data/all_gait.csv", index_col=None)
    train_inds, val_inds, test_inds = train_val_test_split_inds(data, train_args.task)

    # Save the test indices
    with open("save/" + ovr_save_pth + "/test_inds.pkl", "wb") as f:
        pickle.dump((test_inds), f)

    for fold, train_split, val_split in zip(range(train_args.folds),
                                            train_inds, val_inds):

        fold_save_pth = f"fold_{fold}"
        if not os.path.exists("save/" + ovr_save_pth + fold_save_pth):
            os.mkdir("save/" + ovr_save_pth + fold_save_pth)

        log = open(f"save/{ovr_save_pth}/{fold_save_pth}/training.log", "w")

        log_list = [log, ovr_log]

        # Save the indices
        with open("save/" + ovr_save_pth + fold_save_pth + "/inds.pkl",
                  "wb") as f:
            pickle.dump((train_split, val_split), f)
        # Load data
        train_data = ATAXIA(train_split, train_args.task)
        val_data = ATAXIA(val_split, train_args.task)
        test_data = ATAXIA(test_inds, train_args.task)

        # print distribution of labels in the test set.
        to_print = f"Distribution of labels in the test set : {np.unique(test_data.labels, return_counts=True)}\n"
        print_and_log(to_print, log_list)

        # print distribution of labels in the test set.
        to_print = f"Distribution of labels in the Val set : {np.unique(val_data.labels, return_counts=True)}\n"
        print_and_log(to_print, log_list)

        # Create the dataloaders
        train_loader = DataLoader(train_data,
                                  batch_size=train_args.batch_size,
                                  shuffle=True)
        val_loader = DataLoader(val_data,
                                batch_size=train_args.batch_size,
                                shuffle=False)
        test_loader = DataLoader(test_data,
                                 batch_size=train_args.batch_size,
                                 shuffle=False)

        # Load the model
        if model_args.layer_num == -2:
            model = MLP(task=train_args.task)
        else:
            model = TruncatedModel(model_args.layer_num,
                                   model_args.use_mlp,
                                   task=train_args.task)
            state_dict = torch.load(model_args.ckpt_path)
            model.load_state_dict(
                state_dict, strict=False
            )  # strict=False because we are loading a subset of the model
        if torch.cuda.is_available():
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
                       config={
                           **vars(train_args),
                           **vars(model_args)
                       },
                       name=str(fold),
                       group=ovr_save_pth)

        best_model = None
        patience = train_args.patience
        best_acc = 0.0
        best_mae = float('inf')
        
        acc, f1, auc, mse, mae, pearson = None, None, None, None, None, None

        # Training loop
        for epoch in range(train_args.epochs):

            mean_loss = train_step(model, train_loader, optimizer, criterion, train_args.task)

            if train_args.with_tracking:
                wandb.log({"Train Loss": mean_loss}, step=epoch)

            if epoch % train_args.log_every == 0:
                to_print = f"Fold : {fold}, Epoch : {epoch}, Loss : {mean_loss}\n"
                print_and_log(to_print, log_list)

            if epoch % train_args.save_every == 0:
                torch.save(
                    model.state_dict(),
                    f"save/{ovr_save_pth}/{fold_save_pth}/model_{epoch}.pth",
                )

            if epoch % train_args.eval_every == 0:

                preds, labels, val_loss = validate(model, val_loader, train_args.task)
                if train_args.task == "classification":
                    acc, f1, auc = evaluate(preds, labels, train_args.task)
                    to_print = f"Fold : {fold}, Epoch : {epoch}, Validation Accuracy : {acc}, Validation F1 : {f1}, Validation AUC : {auc}\n"
                else:
                    mse, mae, pearson = evaluate(preds, labels,
                                                 train_args.task)
                    to_print = f"Fold : {fold}, Epoch : {epoch}, Validation MSE : {mse}, Validation MAE : {mae}, Validation Pearson : {pearson}\n"
                print_and_log(to_print, log_list)
                if train_args.with_tracking:
                    if train_args.task == "classification":
                        wandb.log(
                            {
                                "Validation Accuracy": acc,
                                "Validation F1": f1,
                                "Validation AUC": auc,
                                "Validation Loss": val_loss,
                            },
                            step=epoch,
                        )
                    else:
                        wandb.log(
                            {
                                "Validation MSE": mse,
                                "Validation MAE": mae,
                                "Validation Pearson": pearson,
                                "Validation Loss": val_loss,
                            },
                            step=epoch,
                        )

                if (acc != None and acc >= best_acc) or (mae != None and mae <= best_mae):  # >= because usually, a later match will have lower loss.
                    patience = train_args.patience
                    best_acc = acc
                    best_mae = mae
                    if train_args.task == "classification":
                        best_model = {
                            "model": model.state_dict(),
                            "epoch": epoch,
                            "f1": f1,
                            "acc": acc,
                            "auc": auc,
                        }
                        to_print = f"Best model saved at epoch {epoch} with Accuracy of {acc}\n"
                    else:
                        best_model = {
                            "model": model.state_dict(),
                            "epoch": epoch,
                            "mae": mae,
                            "mse": mse,
                            "pearson": pearson,
                        }
                        to_print = f"Best model saved at epoch {epoch} with MAE of {mae}\n"
                    torch.save(
                        best_model,
                        f"save/{ovr_save_pth}/{fold_save_pth}/best_model.pth",
                    )
                    print_and_log(to_print, log_list)
                else:
                    patience -= 1
                    if patience == 0:
                        to_print = "EXITING THROUGH AN EARLY STOP.\n"
                        print_and_log(to_print, log_list)
                        break

        # Final evaluation on the best model
        state_dict = torch.load(
            f"save/{ovr_save_pth}/{fold_save_pth}/best_model.pth")["model"]
        model.load_state_dict(state_dict)

        preds, labels, _ = validate(model, test_loader, train_args.task)
        if train_args.task == "classification":
            acc, f1, auc = evaluate(preds, labels, train_args.task)
            to_print = f"Fold : {fold}, Final Evaluation\nTest Accuracy : {acc}, Test F1 : {f1}, Test AUC : {auc}\n"
        else:
            mse, mae, pearson = evaluate(preds, labels, train_args.task)
            to_print = f"Fold : {fold}, Final Evaluation\nTest MSE : {mse}, Test MAE : {mae}, Test Pearson : {pearson}\n"
        print_and_log(to_print, log_list)
        log.close()
        if train_args.with_tracking:
            wandb.finish()

        if train_args.task == "classification":
            ovr_results["Test Accuracy"].append(acc)
            ovr_results["Test F1"].append(f1)
            ovr_results["Test AUC"].append(auc)
            ovr_results["Test Positives"].append(sum(labels))
            ovr_results["Test Negatives"].append(len(labels) - sum(labels))
            ovr_results["Pred Positives"].append(sum(preds))
            ovr_results["Pred Negatives"].append(len(preds) - sum(preds))
        else:
            ovr_results["Test MSE"].append(mse)
            ovr_results["Test MAE"].append(mae)
            ovr_results["Test Pearson"].append(pearson)

    # Create a dataframe and save the results in a csv
    results = pd.DataFrame(ovr_results)
    results.to_csv(f"save/{ovr_save_pth}/results.csv")

    # Average the results and print them
    if train_args.task == "classification":
        mean_acc = results["Test Accuracy"].mean()
        mean_f1 = results["Test F1"].mean()
        mean_auc = results["Test AUC"].mean()
        to_print = f"Mean Test Accuracy : {mean_acc}, Mean Test F1 : {mean_f1}, Mean Test AUC : {mean_auc}\n"
    else:
        mean_mse = results["Test MSE"].mean()
        mean_mae = results["Test MAE"].mean()
        mean_pearson = results["Test Pearson"].mean()
        to_print = f"Mean Test MSE : {mean_mse}, Mean Test MAE : {mean_mae}, Mean Test Pearson : {mean_pearson}\n"
    print(to_print, end="")
    ovr_log.write(to_print)
    ovr_log.close()  # .close() will anyways flush


if __name__ == "__main__":
    main()
