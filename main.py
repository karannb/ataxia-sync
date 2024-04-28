import os
import random
import numpy as np
import pandas as pd
from typing import Tuple
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ATAXIA, splitter
from model.st_gcn import TruncatedModel


class TrainArgs:
    with_tracking: bool = False
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


def seed_all(seed: int):
    """Setup random state from a seed for `torch`, `random` and optionally `numpy` (if can be imported).

    Args:
        seed: Random state seed
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def parse_args() -> Tuple[TrainArgs, ModelArgs]:

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
    parser.add_argument("-b", "--batch_size", type=int, default=256, help="Select Batch Size.")
    parser.add_argument(
        "-e", "--epochs", type=int, default=1000, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Select Learning Rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight Decay for all parameters."
    )
    parser.add_argument(
        "--folds", type=int, default=10, help="Number of folds for cross validation."
    )
    parser.add_argument(
        "--eval_every", type=int, default=10, help="Evaluate every eval_every epochs."
    )
    parser.add_argument(
        "--save_every", type=int, default=50, help="Save model every save_every epochs."
    )
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Seeds the experiment.")

    # Model params
    parser.add_argument(
        "--layer_num", type=int, default=4, help="Decides which block of STGCN is to be used."
    )
    parser.add_argument(
        "--use_mlp",
        default=False,
        action="store_true",
        help="Use a MLP instead of a Conv2d.",
    )
    parser.add_argument(
        "--ensemble",
        default=False,
        action="store_true",
        help="Will do an ensemble of 5 heads when True.",
    )
    parser.add_argument(
        "--ckpt_path",
        default="ckpts/st_gcn.kinetics.pt",
        help="Path to the checkpoint file.",
    )

    args = parser.parse_args()

    train_args = TrainArgs()
    train_args.with_tracking = args.with_tracking
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
    model_args.ensemble = args.ensemble
    model_args.ckpt_path = args.ckpt_path

    return train_args, model_args


def validate(model: nn.Module, loader: DataLoader) -> tuple:

    model.eval()
    preds = []
    labels = []
    loss = []
    for i, (X, y) in enumerate(loader):

        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()

        features = F.softmax(model(X), dim=-1)
        test_loss = F.cross_entropy(features, y)
        loss.append(test_loss.item())
        preds.append(features.argmax(dim=1).cpu().numpy())
        labels.append(y.cpu().numpy())
    
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return preds, labels, sum(loss) / len(loss)


def main():

    # Parse the arguments
    train_args, model_args = parse_args()
    if train_args.with_tracking:
        import wandb
    ovr_results = {"Test Accuracy": [], "Test F1": [], "Test AUC": []}

    ovr_save_pth = f"epoch_{train_args.epochs}_seed_{train_args.seed}_lr_{train_args.lr}_wd_{train_args.weight_decay}_folds_{train_args.folds}_layer_{model_args.layer_num}_mlp_{model_args.use_mlp}_ensemble_{model_args.ensemble}/"
    if not os.path.exists("save/" + ovr_save_pth):
        os.mkdir("save/" + ovr_save_pth)
        
    ovr_log = open(f"save/{ovr_save_pth}/ovr.log", "w")

    for fold in range(train_args.folds):

        fold_save_pth = f"fold_{fold}"
        if not os.path.exists("save/" + ovr_save_pth + fold_save_pth):
            os.mkdir("save/" + ovr_save_pth + fold_save_pth)

        log = open(f"save/{ovr_save_pth}/{fold_save_pth}/training.log", "w")

        # Load the data
        train_inds, test_inds = splitter(149)  # FIXME : Hardcoded
        train_data = ATAXIA(train_inds)
        test_data = ATAXIA(test_inds)

        train_loader = DataLoader(
            train_data, batch_size=train_args.batch_size, shuffle=True
        )
        test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

        if model_args.ensemble:
            raise NotImplementedError("Ensemble not implemented yet.")

        # Load the model
        model = TruncatedModel(model_args.layer_num, model_args.use_mlp)
        state_dict = torch.load(model_args.ckpt_path)
        model.load_state_dict(
            state_dict, strict=False
        )  # strict=False because we are loading a subset of the model
        if torch.cuda.is_available():
            model = model.to("cuda:0")

        # Define the optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=train_args.lr, weight_decay=train_args.weight_decay
        )

        # W&B
        if train_args.with_tracking:
            wandb.init(
                project="ataxia",
                config={**vars(train_args), **vars(model_args)},
                name=str(fold),
                group=ovr_save_pth
            )

        best_model = None
        patience = train_args.patience
        best_f1 = 0.0

        # Training loop
        for epoch in range(train_args.epochs):
            model.train()
            losses = []
            for _, (X, y) in enumerate(train_loader):
                if torch.cuda.is_available():
                    X = X.cuda()
                    y = y.cuda()
                features = F.softmax(model(X), dim=-1)
                loss = criterion(features, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if train_args.with_tracking:
                wandb.log({"Train Loss": sum(losses) / len(losses)}, step=epoch)

            if epoch % train_args.log_every == 0:
                to_print = f"Fold : {fold}, Epoch : {epoch}, Loss : {sum(losses)/len(losses)}\n"
                print(to_print, end="")
                log.write(to_print)
                ovr_log.write(to_print)

            if epoch % train_args.save_every == 0:
                torch.save(
                    model.state_dict(),
                    f"save/{ovr_save_pth}/{fold_save_pth}/model_{epoch}.pth",
                )

            if epoch % train_args.eval_every == 0:

                preds, labels, test_loss = validate(model, test_loader)
                acc = accuracy_score(labels, preds)
                f1 = f1_score(labels, preds)
                auc = roc_auc_score(labels, preds)
                to_print = f"Fold : {fold}, Epoch : {epoch}, Test Accuracy : {acc}, Test F1 : {f1}, Test AUC : {auc}\n"
                print(to_print, end="")
                log.write(to_print)
                if train_args.with_tracking:
                    wandb.log(
                        {
                            "Test Accuracy": acc,
                            "Test F1": f1,
                            "Test AUC": auc,
                            "Test Loss": test_loss,
                        },
                        step=epoch,
                    )

                if f1 > best_f1:
                    patience = train_args.patience
                    best_f1 = f1
                    best_model = {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "f1": f1,
                        "acc": acc,
                        "auc": auc,
                    }
                    torch.save(
                        best_model,
                        f"save/{ovr_save_pth}/{fold_save_pth}/best_model.pth",
                    )
                    log.write(
                        f"Best model saved at epoch {epoch} with F1 score of {f1}\n"
                    )
                    ovr_log.write(
                        f"Best model saved at epoch {epoch} with F1 score of {f1}\n"
                    )
                else:
                    patience -= 1
                    if patience == 0:
                        to_print = "EXITING THROUGH AN EARLY STOP.\n"
                        print(to_print, end="")
                        log.write(to_print)
                        ovr_log.write(to_print)
                        break

        # Final evaluation on the best model
        state_dict = torch.load(f"save/{ovr_save_pth}/{fold_save_pth}/best_model.pth")[
            "model"
        ]
        model.load_state_dict(state_dict)

        preds, labels, test_loss = validate(model, test_loader)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        auc = roc_auc_score(labels, preds)
        to_print = f"Final Evaluation\nTest Accuracy : {acc}, Test F1 : {f1}, Test AUC : {auc}\n"
        print(to_print, end="")
        log.write(to_print)
        ovr_log.write(to_print)
        log.close()
        wandb.finish()

        ovr_results["Test Accuracy"].append(acc)
        ovr_results["Test F1"].append(f1)
        ovr_results["Test AUC"].append(auc)

    results = pd.DataFrame(ovr_results)
    results.to_csv(f"save/{ovr_save_pth}/results.csv")
    mean_acc = results["Test Accuracy"].mean()
    mean_f1 = results["Test F1"].mean()
    mean_auc = results["Test AUC"].mean()
    to_print = f"Mean Test Accuracy : {mean_acc}, Mean Test F1 : {mean_f1}, Mean Test AUC : {mean_auc}\n"
    print(to_print, end="")
    ovr_log.write(to_print)
    ovr_log.close()

if __name__ == "__main__":
    main()
