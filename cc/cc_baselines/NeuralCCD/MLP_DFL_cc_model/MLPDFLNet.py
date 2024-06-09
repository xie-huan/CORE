import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net1(nn.Module):
    def __init__(self, input_dim):
        super(Net1, self).__init__()
        self.input_dim = input_dim
        hidden_dim = math.floor(self.input_dim * 1.5)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc2(self.relu(self.fc1(x))))
        # x = self.fc2(self.relu(self.fc1(x)))
        return x


class Net2(nn.Module):
    def __init__(self, input_dim):
        super(Net2, self).__init__()
        self.input_dim = input_dim
        hidden_dim = math.floor(self.input_dim * 1.5)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.fc2(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(self.relu(self.fc1(x))))
        return x


class Net3(nn.Module):
    def __init__(self, input_dim):
        super(Net3, self).__init__()
        self.input_dim = input_dim
        hidden_dim = math.floor(self.input_dim * 1.5)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc2(self.relu(self.fc1(x))))
        # x = self.fc2(self.relu(self.fc1(x)))
        return x


class Net4(nn.Module):
    def __init__(self, input_dim):
        super(Net4, self).__init__()
        self.input_dim = input_dim
        hidden_dim_1 = math.floor(self.input_dim * 1.5)
        hidden_dim_2 = math.floor(hidden_dim_1 * 1.5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_1)
        self.fc4 = nn.Linear(hidden_dim_1, input_dim)
        self.fc5 = nn.Linear(input_dim, 2)

    def forward(self, x):
        x = self.fc5(self.relu(self.fc4(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))))))
        # x=self.fc5(x)
        return x


class MLPDFLnet(nn.Module):
    def __init__(self, embeddingnet1, embeddingnet2, embeddingnet3, embeddingnet4):
        super(MLPDFLnet, self).__init__()
        self.embeddingnet1 = embeddingnet1
        self.embeddingnet2 = embeddingnet2
        self.embeddingnet3 = embeddingnet3
        self.embeddingnet4 = embeddingnet4
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, ssp, cr, sf):
        embedded_ssp = self.embeddingnet1(ssp)
        embedded_cr = self.embeddingnet2(cr)
        embedded_sf = self.embeddingnet3(sf)

        embedded_input_layer = torch.hstack((embedded_ssp, embedded_cr, embedded_sf))
        prob = self.embeddingnet4(embedded_input_layer)

        prob = self.softmax(prob)
        return prob
