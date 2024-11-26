import torch
import torch.nn as nn
from config import Config

class FN_Net(nn.Module):
    def __init__(self, input_dim, output_dim, n_neurons, n_hidden_layers=1, dropout_rate=0.01):
        super(FN_Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        layers.append(nn.Linear(input_dim, n_neurons))
        layers.append(nn.Tanh())
        
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout_rate))
            
        layers.append(nn.Linear(n_neurons, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)