import torch
import torch.nn as nn

class MLP(nn.Module):
    
    def __init__(self, hid_dim=512):
        
        super().__init__()
        
        self.hid_dim = hid_dim
        self.mlp = nn.Sequential(*[
            nn.Linear(75*18*3, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 2)
        ])
        
    def forward(self, x:torch.tensor):
        
        bs = x.shape[0]
        x = x.reshape(bs, -1) #(bs, 75*18*3)
        y_hat = self.mlp(x) #(bs, 2)
        y_hat = y_hat.reshape(bs, -1)
        
        return y_hat