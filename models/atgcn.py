"""
This code has been adapted from https://github.com/yysijie/st-gcn 
to match the truncation discussed in section 3.4 of the paper.
"""

import sys
sys.path.append("./")

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.st_gcn import Model
from models.utils.graph import Graph
from models.ResGCNv1.nets import ResGCN
from models.ResGCNv1.modules import ResGCN_Module
from models.utils.pooler import custom_pool_2d

# Number of output channels for each layer
# used to determine the number of features to be extracted
LAYER2DIM = {
    0: 64,
    1: 64,
    2: 64,
    3: 64,
    4: 128,
    5: 128,
    6: 128,
    7: 256,
    8: 256,
    9: 256,
    -1: 256,
    "resgcn": 128
}

class TruncatedSTGCN(Model):
    r"""Truncated STGCN which uses pretrained weights from the original model (recommended),
    and return activations till a certain layer.

    Args:
        layer (int): The layer till which the activations are to be returned.

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self,
                 layer=4,
                 use_mlp=False,
                 task="classification",
                 freeze_encoder: bool = False,
                 **kwargs):
        '''
        Args:
        ----
        layer: int
            The layer till which the activations are to be returned.
        use_mlp: bool
            If True, uses a MLP head, else uses a Conv2d head.
        task: str
            The task to be performed. Either "classification" or "regression".
        freeze_encoder: bool
            If True, freezes the encoder (pretrained STGCN) weights.
        '''

        super().__init__(
            in_channels=3,
            num_class=400,
            graph_args={
                "layout": "openpose",
                "strategy": "spatial"
            },
            edge_importance_weighting=False if freeze_encoder else True,
            return_hidden_states=True,
        )
        self.layer = layer

        # Freeze the encoder
        if freeze_encoder:
            self.st_gcn_networks.requires_grad_(False)
            self.fcn.requires_grad_(False)

        # freeze unused parts of the encoder
        for i, net in enumerate(self.st_gcn_networks):
            if (i+1 > layer) and (layer != -1):
                net.requires_grad_(False)
        
        # -1 => full encoder + FCN
        if layer != -1:
            self.fcn.requires_grad_(False)

        # Define the head
        if task == "classification":
            num_class = 2
        elif task == "regression":
            num_class = 1
        else:
            raise NotImplementedError

        # deepnet is just a bigger head
        if use_mlp:
            self.head = nn.Linear(LAYER2DIM[layer], num_class)
        else:
            self.head = nn.Conv2d(LAYER2DIM[layer],
                                    num_class,
                                    kernel_size=1)

    def forward(self, x: torch.Tensor):

        # Forward pass on the STGCN
        x, hidden_states = super().forward(x)

        # Extract features from the desired layer
        if isinstance(self.layer, int):
            features = custom_pool_2d(hidden_states[self.layer], x.size(0), 1)
        else:
            features = x.flatten(1)

        preds = self.head(features)
        preds = preds.view(x.size(0), -1)  # Flatten the output

        return preds


class TruncatedResGCN(ResGCN):
    r"""Truncated ResGCN which uses pretrained weights from the original model (recommended),
    and return activations till a certain layer.
    
    Args:
        layer (int): The layer till which the activations are to be returned.
    """

    def __init__(self, 
                 layer=4, 
                 task="classification",
                 freeze_encoder: bool = False,
                 **kwargs):
        kwargs.update({'module': ResGCN_Module, 
                       'attention': None, 
                       'structure': [1, 2, 2, 2], 
                       'block': 'Bottleneck', 
                       'reduction': 8})
        graph = Graph(layout='resgcn', strategy='distance', max_hop=3)
        model_args = {
            "A": torch.tensor(graph.A, dtype=torch.float32, requires_grad=False),
            "num_class": 128,
            "num_input": 1,
            "num_channel": 3,
            "parts": graph.parts,
        }
        super().__init__(**model_args, **kwargs)

        # freeze the encoder if required
        if freeze_encoder:
            for param in self.parameters():
                param.requires_grad = False

        # define a head
        if task == "classification":
            self.head = nn.Linear(LAYER2DIM["resgcn"], 2)
        elif task == "regression":
            self.head = nn.Linear(LAYER2DIM["resgcn"], 1)
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor):
        x = x.squeeze(2)
        x = super().forward(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

if __name__ == "__main__":
    # # Test the TruncatedSTGCN model
    # model = TruncatedSTGCN(layer=4)
    # x = torch.randn(1, 3, 300, 25, 2)
    # out = model(x)
    # print(out.size())

    # Test the TruncatedResGCN model
    model = TruncatedResGCN()
    x = torch.randn(32, 1, 3, 60, 17)
    out = model(x)
    print(out.size())