import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPAttacker(nn.Module):
    # define nn
    def __init__(self, args, leak='posterior'):
        super(MLPAttacker, self).__init__()
        if leak == 'posterior':
            input_dim = 2
        elif leak == 'repr':
            input_dim = int(args.hidden_dim)

        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        X = self.softmax(X)
        return X