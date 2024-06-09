import math

import torch
from torch import nn


class CoverageInfoSematicNet(nn.Module):
    def __init__(self, input_dim):
        super(CoverageInfoSematicNet, self).__init__()
        self.input_dim = input_dim
        hidden_dim_1 = math.floor(self.input_dim * 1.5)
        hidden_dim_2 = math.floor(hidden_dim_1 * 1.5)
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.drop_out_1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.drop_out_2 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_1)
        self.drop_out_3 = nn.Dropout(p=0.25)
        self.fc4 = nn.Linear(hidden_dim_1, input_dim)
        self.drop_out_4 = nn.Dropout(p=0.25)
        self.fc5 = nn.Linear(input_dim, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        temp_x = self.fc1(x)
        temp_x = self.drop_out_1(temp_x)
        temp_x = self.fc2(temp_x)
        temp_x = self.drop_out_2(temp_x)
        temp_x = self.fc3(temp_x)
        temp_x = self.drop_out_3(temp_x)
        temp_x = self.fc4(temp_x)
        temp_x = self.drop_out_4(temp_x)
        x = self.fc5(temp_x)
        x = self.softmax(x)
        return x


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


class EFCNetwork(nn.Module):
    def __init__(self, embeddingnet1, embeddingnet2, embeddingnet3, embeddingnet4):
        super(EFCNetwork, self).__init__()
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


class EFCNetwork2(nn.Module):
    def __init__(self, ssp_net, cr_net, sf_net, combined_net):
        super(EFCNetwork2, self).__init__()
        # expert combined
        self.ssp_net = ssp_net
        self.cr_net = cr_net
        self.sf_net = sf_net
        self.combined_net = combined_net
        self.softmax = nn.Softmax(dim=1)

    def forward(self, ssp, cr, sf):
        ssp = self.ssp_net(ssp)
        cr = self.cr_net(cr)
        sf = self.sf_net(sf)

        embedded_input_layer = torch.hstack((ssp, cr, sf))
        prob = self.combined_net(embedded_input_layer)
        prob = self.softmax(prob)
        return prob


class ExpertFeatureCombinedNetwork(nn.Module):
    def __init__(self, sematic_embedding_net, combined_net):
        super(ExpertFeatureCombinedNetwork, self).__init__()
        # coverage_info_sematic
        self.sematic_embedding_net = sematic_embedding_net

        # expert combined
        self.combined_net = combined_net

        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, ssp, cr, sf):
        coverage_info = self.sematic_embedding_net(x)
        expert_info = self.combined_net(ssp, cr, sf)

        combined = torch.cat((coverage_info, expert_info), dim=1)

        combined = self.fc1(combined)
        combined = self.relu(combined)
        prob = self.fc2(combined)
        # prob = self.combined_net(x)
        prob = self.softmax(prob)
        return prob


class EFCNetworkWithoutCovInfo(nn.Module):
    def __init__(self, ssp_net, cr_net, sf_net, combined_net, size):
        super(EFCNetworkWithoutCovInfo, self).__init__()
        # expert combined
        self.ssp_net = ssp_net
        self.cr_net = cr_net
        self.sf_net = sf_net
        self.combined_net = combined_net
        self.size = size

        self.fc = nn.Linear(self.size, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, ssp, cr, sf):
        ssp = self.ssp_net(ssp)
        cr = self.cr_net(cr)
        sf = self.sf_net(sf)

        combined = torch.cat((ssp, cr, sf), dim=1)
        combined = self.combined_net(combined)

        prob = self.fc(combined)
        prob = self.softmax(prob)
        return prob


class EFCNetworkWithoutExpertFeature(nn.Module):
    def __init__(self, sematic_embedding_net, size):
        super(EFCNetworkWithoutExpertFeature, self).__init__()
        # coverage_info_sematic
        self.sematic_embedding_net = sematic_embedding_net
        self.size = size
        self.fc = nn.Linear(self.size, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.sematic_embedding_net(x)

        prob = self.fc(x)
        prob = self.softmax(prob)
        return prob
