import torch
import torch.nn as nn
from torch.optim import Adam
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class backpropnetwork(torch.nn.Module):

    def __init__(self, input_size, output_size, dropout=0.2):

        super(backpropnetwork, self).__init__()
        self.dropout=dropout

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 2000, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(2000, 2000, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(2000, 2000, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(2000, 2000, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(2000, output_size),
        )      
        
    def forward(self, x):
        out = self.model(x)
        return out