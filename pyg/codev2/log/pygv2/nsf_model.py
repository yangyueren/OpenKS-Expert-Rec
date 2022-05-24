
from collections import defaultdict
import os
import copy
from turtle import pd
from tqdm import tqdm
import torch
import torch.nn  as nn
import torch.nn.functional  as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
from util import save_to_disk

from util import load_from_disk
from mylogger import logger

from torch_geometric.nn import GATConv, Linear, to_hetero
from torch_geometric.nn import HGTConv, Linear

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x

class HGT(torch.nn.Module):
    def __init__(self, data, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
        self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict

class GraphModel(nn.Module):
    def __init__(self, config, G):
        super(GraphModel, self).__init__()
        self.config = config
        # import pdb; pdb.set_trace()
        hidden_channels = config.hidden_channels
        out_channels = config.out_channels
        graph = GAT(hidden_channels=hidden_channels, out_channels=out_channels)
        Gg = copy.deepcopy(G)
        del Gg['project']
        self.graph = to_hetero(graph, Gg.metadata(), aggr='sum')
        # self.graph = HGT(G, hidden_channels=hidden_channels, out_channels=out_channels, num_heads=2, num_layers=2)
        self.fc1 = nn.Linear(out_channels*2, out_channels)
        self.drop = nn.Dropout()
        self.fc2 = nn.Linear(out_channels, 1)
        self.fc3 = nn.Linear(768, out_channels)


    def get_graph_emb(self, sub_G, nodeids, node_type_name):
        if node_type_name == 'project':
            update_G = sub_G
            nodes_feat = self.fc3(update_G[node_type_name].x)
        else:
            update_G = self.graph(sub_G.x_dict, sub_G.edge_index_dict)
            nodes_feat = update_G[node_type_name]
        idxs = []
        nids = sub_G[node_type_name].n_id
        assert len(nids.shape) == 1, 'shape error'
        for nodeid in nodeids:
            idx = (nids == nodeid).nonzero(as_tuple=True)[0]
            idxs.append(idx)
        idxs = torch.tensor(idxs).squeeze()
        assert len(idxs) == len(nodeids), 'error'
        feat = nodes_feat[idxs]
        return feat

    def forward(self, person_sub_graph, person, project_sub_graph, project):
        # import pdb;pdb.set_trace()
        person_emb = self.get_graph_emb(person_sub_graph, person, 'person')
        project_emb = self.get_graph_emb(project_sub_graph, project, 'project')

        person_emb = person_emb.reshape(project_emb.shape[0], -1, person_emb.shape[-1])
        
        mix = torch.cat([person_emb, project_emb.unsqueeze(1).repeat(1, person_emb.shape[1], 1)], dim=-1)
        mix = self.fc1(mix)
        mix = F.relu(self.drop(mix))

        logits = self.fc2(mix).squeeze()
        return logits
