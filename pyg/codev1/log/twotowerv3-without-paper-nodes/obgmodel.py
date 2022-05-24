
from torch_geometric.nn import GATConv, Linear, to_hetero
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, to_hetero
import torch
from torch.optim import Optimizer
import torch.nn.functional as F

import yaml
from util import Dict2Obj
from nsf_dataset import NSFDataset
# dataset = OGB_MAG(root='./tmp', preprocess='metapath2vec', transform=T.ToUndirected())
# data = dataset[0]
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader

transform = T.ToUndirected()  # Add reverse edge types.
data = OGB_MAG(root='../tmp', preprocess='metapath2vec', transform=transform)[0]
print(data.has_isolated_nodes())

# data['author'].x.requires_grad = True
# data['paper'].x.requires_grad = True
# data['field_of_study'].x.requires_grad=True
# import copy

# aa = copy.copy(data['paper'].x)
# import pdb;pdb.set_trace()

# with open('./config.yml', 'r') as f:
#         config = yaml.safe_load(f)
#         config = Dict2Obj(config).config
# train_dataset = NSFDataset(config, config.train_data)
# data = train_dataset.get_graph()
# data['paper'].train_mask = torch.ones(len(data['paper'].x)).bool()
# data['paper'].y = torch.ones(len(data['paper'].x))
# print(data.has_isolated_nodes())
# transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])
# data = transform(data)
# print(data)

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x

d = torch.device('cpu')
model = GAT(hidden_channels=64, out_channels=1).to(d)
model = to_hetero(model, data.metadata(), aggr='sum')

model = model.to(d)
data = data.to(d)



train_loader = NeighborLoader(
    data,
    # Sample 15 neighbors for each node and each edge type for 2 iterations:
    num_neighbors=[15] * 2,
    # Use a batch size of 128 for sampling training nodes of type "paper":
    batch_size=128,
    input_nodes=('paper', data['paper'].train_mask),
)

batch = next(iter(train_loader))

def train():
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    total_examples = total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch_size = batch['paper'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        
        loss = F.binary_cross_entropy_with_logits(out['paper'][:batch_size].squeeze(),
                               batch['paper'].y[:batch_size].float())
        loss.backward()
        import pdb; pdb.set_trace()
        optimizer.step()
        print(loss)
        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples
train()