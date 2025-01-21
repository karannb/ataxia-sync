'''
This file is a simple implementation of an MLP layer,
used as a trivial baseline internally.
To run this you can set layer_num to -2.
'''
import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, hid_dim=512, task="classification"):
        """
        Builds a simple one hidden layer MLP.

        Args:
            hid_dim (int, optional): The hidden dimension of the MLP. Defaults to 512.
            task (str, optional): The task to be performed (regression / classification). Defaults to "classification".
        """

        super().__init__()

        self.hid_dim = hid_dim
        if task == "classification":
            num_class = 2
        elif task == "regression":
            num_class = 1
        self.mlp = nn.Sequential(*[
            nn.Linear(75 * 18 * 3, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, num_class)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        bs = x.shape[0]
        x = x.reshape(bs, -1)  #(bs, 75*18*3)
        y_hat = self.mlp(x)  #(bs, num_cls)
        y_hat = y_hat.reshape(bs, -1)

        return y_hat
