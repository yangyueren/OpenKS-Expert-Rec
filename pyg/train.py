
import torch
from torch_geometric.data import Data

from tqdm import tqdm
# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

# data = Data(x=x, edge_index=edge_index)

# from torch_geometric.datasets import TUDataset

# dataset = TUDataset(root='./tmp', name='ENZYMES', use_node_attr=True)
N = 3000000
num_node_features = 768*3
num_classes = 5
x = torch.randn(N, num_node_features)

edge_index = torch.randint(N-1, (2, N*(N/2))).long()
# torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
train_mask = torch.arange(0, int(N*0.5))
val_mask = torch.arange(int(N*0.5), int(N*0.9))
test_mask = torch.arange(int(N*0.9), N)
y = torch.randint(num_classes, (N,))

data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
# from torch_geometric.datasets import Planetoid

# dataset = Planetoid(root='./tmp', name='Cora')

# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv

# class GCN(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = GCNConv(num_node_features, 16)
#         self.conv2 = GCNConv(16, num_classes)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)

#         return F.log_softmax(x, dim=1)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GCN().to(device)
# data = data.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# # import pdb; pdb.set_trace()
# model.train()
# for epoch in tqdm(range(200)):
#     optimizer.zero_grad()
#     out = model(data)
#     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()
# model.eval()
# pred = model(data).argmax(dim=1)
# correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
# acc = int(correct) / int(data.test_mask.sum())
# print(f'Accuracy: {acc:.4f}')


from torch_geometric.nn import GATConv

from torch_geometric.utils import add_self_loops,degree
from torch_geometric.datasets import Planetoid
import ssl
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.gat1=GATConv(num_node_features,8,8,dropout=0.6)
        self.gat2=GATConv(64,num_classes,1,dropout=0.6)

    def forward(self,data):
        x,edge_index=data.x, data.edge_index
        x=self.gat1(x,edge_index)
        x=self.gat2(x,edge_index)
        return F.log_softmax(x,dim=1)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in tqdm(range(100)):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct/int(data.test_mask.sum())
print('Accuracy:{:.4f}'.format(acc))
