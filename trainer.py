import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
import numpy as np
from dataset import ATAXIA
from model.st_gcn import Model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

MAX_LAYERS = 9

LAYER2DIM = {
    0 : 64,
    1 : 64,
    2 : 64,
    3 : 64,
    4 : 128,
    5 : 128,
    6 : 128,
    7 : 256,
    8 : 256,
    9 : 256,
    -1 : 256,
    'all' : 400
}

def custom_pool_2d(x : torch.Tensor, N, M):
    # global pooling
    x = F.avg_pool2d(x, x.size()[2:])
    x = x.view(N, M, -1, 1, 1).mean(dim=1)
    
    return x

if __name__ == '__main__':
        
    data = ATAXIA()
    test_data = ATAXIA(mode="test")
    
    batch_size = 256
    lr=3e-5
    num_epochs = 1000
    decay = 0.0#001
    
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False)
    
    # for layer in (list(range(MAX_LAYERS)) + [-1] + ['all']):
    layer = 4
    # basically consider activations only till layer, and see how the model does
    weights = torch.load("models/st_gcn.kinetics.pt")
    STGCN = Model(in_channels=3, 
                    num_class=400, 
                    edge_importance_weighting=True, 
                    graph_args={"layout": "openpose", "strategy": "spatial"},
                    return_hidden_states=True)
    STGCN.load_state_dict(weights, strict=True)
    STGCN.to("cuda:0")
    # STGCN.eval()
    # for param in STGCN.parameters():
    #     param.requires_grad = False
    if isinstance(layer, int):
        FCNs = {1 : nn.Conv2d(LAYER2DIM[layer], 2, kernel_size=1).to("cuda:0"),
                2 : nn.Conv2d(LAYER2DIM[layer], 2, kernel_size=1).to("cuda:0"),
                3 : nn.Conv2d(LAYER2DIM[layer], 2, kernel_size=1).to("cuda:0"),
                4 : nn.Conv2d(LAYER2DIM[layer], 2, kernel_size=1).to("cuda:0"),
                5 : nn.Conv2d(LAYER2DIM[layer], 2, kernel_size=1).to("cuda:0")}
        # small_MLP = nn.Conv2d(LAYER2DIM[layer], 2, kernel_size=1).to("cuda:0")
    else:
        small_MLP = nn.Linear(LAYER2DIM[layer], 2).to("cuda:0")
    # ovr_params = list(small_MLP.parameters()) + list(STGCN.parameters())
    ovr_params = list(STGCN.parameters()) + list(FCNs[1].parameters()) + list(FCNs[2].parameters()) + list(FCNs[3].parameters()) + list(FCNs[4].parameters()) + list(FCNs[5].parameters())
    optimizer = torch.optim.Adam(ovr_params, lr=lr, weight_decay=decay)
    
    wandb.init(project="ataxia",
                config={"layer" : layer,
                        "batch_size" : batch_size,
                        "lr" : lr,
                        "epochs" : num_epochs,
                        "weight_decay" : decay,})
    
    for epoch in range(num_epochs):
        losses = []
        for i, (X, y) in enumerate(loader):
            X = X.cuda()
            y = y.cuda()
            out, features = STGCN(X)
            if isinstance(layer, int):
                features = custom_pool_2d(features[layer], X.shape[0], 1)
            else:
                features = out
            feats = []
            for i in FCNs:
                feats.append(FCNs[i](features))
            features = sum(feats)/len(feats)
            # features = small_MLP(features)
            features = features.view(features.size(0), -1)
            y_hat = F.softmax(features, dim=-1)
            loss = nn.CrossEntropyLoss()(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        wandb.log({"Train Loss" : sum(losses)/len(losses)},
                    step=epoch)
        if (epoch) % 10 == 0:
            print(f"Layer {layer} | Epoch {epoch} | Train Loss: {sum(losses)/len(losses)}")
            
            # Check accuracy
            correct = 0
            for i, (X, y) in enumerate(loader):
                X = X.cuda()
                y = y.cuda()
                out, features = STGCN(X)
                if isinstance(layer, int):
                    features = custom_pool_2d(features[layer], X.shape[0], 1)
                else:
                    features = out
                # features = small_MLP(features)
                feats = []
                for i in FCNs:
                    feats.append(FCNs[i](features))
                features = sum(feats)/len(feats)
                features = features.view(features.size(0), -1)
                y_hat = torch.argmax(features, dim=-1)
                correct += torch.sum(y_hat == y)
            print(f"Layer {layer} | Epoch {epoch} | Accuracy: {correct / len(data)}")
    
            correct = 0
            preds = []
            lbls = []
            losses = []
            for i, (X, y) in enumerate(test_loader):
                X = X.cuda()
                y = y.cuda()
                out, features = STGCN(X)
                if isinstance(layer, int):
                    features = custom_pool_2d(features[layer], X.shape[0], 1)
                else:
                    features = out
                # features = small_MLP(features)
                feats = []
                for i in FCNs:
                    feats.append(FCNs[i](features))
                features = sum(feats)/len(feats)
                features = features.view(features.size(0), -1)
                y_hat = torch.argmax(features, dim=-1)
                losses.append(nn.CrossEntropyLoss()(F.softmax(features, -1), y).item())
                preds.append(y_hat.cpu().numpy())
                lbls.append(y.cpu().numpy())
            
            preds = np.concatenate(preds)
            lbls = np.concatenate(lbls)
            print(f"Layer {layer} | Epoch {epoch} | Test Accuracy: {accuracy_score(lbls, preds)}")
            print(f"Layer {layer} | Epoch {epoch} | Test F1: {f1_score(lbls, preds)}")
            print(f"Layer {layer} | Epoch {epoch} | Test Precision: {precision_score(lbls, preds)}")
            print(f"Layer {layer} | Epoch {epoch} | Test Recall: {recall_score(lbls, preds)}")
            print(f"Layer {layer} | Epoch {epoch} | Test AUC: {roc_auc_score(lbls, preds)}")
            print(f"Layer {layer} | Epoch {epoch} | Test Loss: {sum(losses)/len(losses)}")
            wandb.log({"Test Accuracy" : accuracy_score(lbls, preds),
                        "Test F1" : f1_score(lbls, preds),
                        "Test Precision": precision_score(lbls, preds),
                        "Test Recall" : recall_score(lbls, preds),
                        "Test AUC": roc_auc_score(lbls, preds),
                        "Test Loss": sum(losses)/len(losses)},
                        step=epoch)
    
    print(f"Done with layer {layer}, Saving....")
    torch.save({
        "STGCN" : STGCN.state_dict(),
        "small_MLP" : small_MLP.state_dict(),
        "layer" : layer,
    }, f"save/{layer}.pth")
    wandb.finish()
    print("Experiment finished.")
        
    print("Done with all layers.")
'''
EXTRA - 

if isinstance(layer, int):
    for name, param in STGCN.named_parameters():
        if name.find('st_gcn_networks') > 0 and (int(name[name.find('st_gcn_networks') + 15]) > layer):
            continue
        elif name.find('edge_importance') > 0 and (int(name[name.find('edge_importance') + 15]) > layer):
            continue
        ovr_params += list(param)
else:
    ovr_params
    
SEEING GRADIENTS OF ANY LAYER - 
print(f"Gradient at unused layers - {STGCN.st_gcn_networks[layer].gcn.conv.weight.grad}")
exit(-1)
'''