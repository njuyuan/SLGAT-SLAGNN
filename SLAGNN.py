import scipy.sparse as sp
import argparse
import torch
from torch.nn import Linear, LayerNorm, ReLU
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, download_url, Data
from openTSNE import TSNE
import matplotlib.pyplot as plt
from time import *
import random
import numpy as np

from PPMIConv import PPMIConv
from util import *
from model import GATConv, AGNNConv
from dataprocessed import *


begin_time = time()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=3)  
parser.add_argument('--epochs', type=int, default=501)
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--no_cuda', action = 'store_true', default=False)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--early_stopping', type=int, default=100)
parser.add_argument('--max_layer', type=int, default=2)
parser.add_argument('--hid', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.8)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--cuda_device', type=int, default=0)
parser.add_argument("--seed", type=int,default=128)
parser.add_argument('--model', type=str, default='slagnn')
parser.add_argument('--lam', type=float, default=1.0) 
parser.add_argument('--k', type=float, default=0.1) 
parser.add_argument('--len', type=int, default=5) 
parser.add_argument('--sample', type=int, default=20) 


args = parser.parse_args()
seed = args.seed
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.cuda_device)

random.seed(seed) 
np.random.seed(seed)
torch.manual_seed(seed)

name = args.dataset
dataset = get_dataset(name)
data = dataset[0]

#data = random_coauthor_amazon_splits(data, dataset.num_classes, args.sample, None) 
#data = random_planetoid_splits(data, dataset.num_classes, args.sample, None)

"""
feature_matrix = cos(data.x)
#feature = get_binarized_kneighbors_graph(data.x, 0)
feature_matrix = get_clipped_matrix(feature_matrix , args.k)
#feature_matrix = get_top_k_matrix(feature_matrix , args.len)
print("edge_index: ", data.edge_index.shape, data.edge_index)
data.edge_index, data.edge_weight = create_data(data, feature_matrix)
print("edge_index1: ", data.edge_index.shape, data.edge_index)

print("edge_weight: ", data.edge_weight.shape, data.edge_weight)

"""

ppmi_edge_index, ppmi_weight = PPMIConv.norm_update(data.edge_index, data.x.shape[0], args.len)
ppmi_edge_index = ppmi_edge_index.cpu().detach().numpy()
ppmi_weight = ppmi_weight.cpu().detach().numpy()
ppmi_matrix = sp.coo_matrix((ppmi_weight, (ppmi_edge_index[0], ppmi_edge_index[1])), shape=(data.x.shape[0], data.x.shape[0]))
ppmi_matrix = ppmi_matrix.toarray()
#print("ppmi: ", ppmi_matrix)
#ppmi_matrix = get_top_k_matrix(ppmi_matrix, args.k)
ppmi_matrix = get_clipped_matrix(ppmi_matrix, args.k) #计算得到ppmi矩阵

test_matrix = ppmi_matrix
test_matrix = sp.coo_matrix(test_matrix) # 基于PPMI矩阵

#print("ppmi_k: ", test_matrix.shape, test_matrix)


#It should be noted that the six standard datasets have graph structures. 
#To reduce the time complexity, we do not use metric learning, 
#only calculate the PPMI matrix to modify the graph structure, which is the special case of our models.

#print("edge_index: ", data.edge_index.shape, data.edge_index)
data.edge_index, data.edge_weight = create_data(data, ppmi_matrix)
#data.edge_index, data.edge_weight = create_data(data)
#print("edge_index1: ", data.edge_index.shape, data.edge_index)



# GAT
class GAT(torch.nn.Module):
    def __init__(self):
        self.hidden = []
        super(GAT, self).__init__()
        self.max_layer = args.max_layer
        self.conv1 = GATConv(dataset.num_features, args.hid, heads=8, dropout=args.dropout, edge_weight=data.edge_weight, lam = args.lam)
        for i in range(self.max_layer - 2):     
            self.hidden.append(GATConv(args.hid*8, args.hid, heads=8, dropout=args.dropout, edge_weight=data.edge_weight, lam = args.lam))
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(args.hid * 8, dataset.num_classes, heads=1, concat=False,
                             dropout=args.dropout, edge_weight=data.edge_weight, lam = args.lam)
     #   self.weight1 = Parameter(torch.Tensor(channels, channels))
        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        for i in range(self.max_layer - 2):
            self.hidden[i].reset_parameters()

    def forward(self):

        x = F.dropout(data.x, p=args.dropout, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        for i in range(self.max_layer-2):
            x = F.dropout(x, p=args.dropout, training=self.training)
            x = F.elu(self.hidden[i](x, data.edge_index))

        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, data.edge_index)
        return x#F.log_softmax(x, dim=1)

class AGNN(torch.nn.Module):
    def __init__(self):
        super(AGNN, self).__init__()
        self.lin1 = torch.nn.Linear(dataset.num_features, args.hid)
        self.prop1 = AGNNConv(requires_grad=False, edge_weight=data.edge_weight, lam = args.lam)
        self.prop2 = AGNNConv(requires_grad=True, edge_weight=data.edge_weight, lam = args.lam)
        self.lin2 = torch.nn.Linear(args.hid, dataset.num_classes)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()
        self.prop2.reset_parameters()


    def forward(self):
        x = F.dropout(data.x, p=args.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, data.edge_index)
        x = self.prop2(x, data.edge_index)
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.lin2(x)
        return x#F.log_softmax(x, dim=1)



if args.model =='slgat':
    model, data = GAT().to(device), data.to(device)

if args.model =='slagnn':
    model, data = AGNN().to(device), data.to(device)

print("model: ", args.model)

print(device)

def train():

    model.train()

    optimizer.zero_grad()
  #  print(data.train_mask)
    preds = model()[data.train_mask]
    preds = F.log_softmax(preds, dim=1)

    loss = F.nll_loss(preds, data.y[data.train_mask])
    loss.backward()

    optimizer.step()
    return float(loss)

    
@torch.no_grad()
def test():

    model.eval()
    embed = model()
    logits = F.log_softmax(embed, dim=1)
    accs = []

    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]       
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


test_accs = []
best_test_accs = []
for run in range(1, args.runs):
    print('')
    print(f'Run {run:02d}:')
    print('')

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0
    best_test_acc = 0
    fo = open('{}{}.txt'.format(args.dataset, args.model), "a+")
    for epoch in range(1, args.epochs):
        loss = train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc

        if tmp_test_acc > best_test_acc:
            best_test_acc = tmp_test_acc

        if epoch%10 == 0:
            print('Epoch: {:03d}, Train Acc: {:.4f}, '
          'Val Acc: {:.4f}, Test Acc: {:.4f}'.format(epoch, train_acc, val_acc, test_acc))

            fo.write("Epoch: "+str(epoch)+'\t'+"Train Acc: "+str(train_acc)+'\t'+"Val Acc: "+str(val_acc)+'\t'+"Test Acc: "+str(test_acc)+'\n')
    fo.close()  
    test_accs.append(test_acc)
    best_test_accs.append(best_test_acc)


test_acc = torch.tensor(test_accs)
print(test_acc)
print('============================')
print(f'Real test results under best validation (Paper report): {test_acc.mean():.4f} ± {test_acc.std():.4f}')

print('============================')
print(best_test_accs)
best_test_acc = torch.tensor(best_test_accs)
print(f'Best test results (Upper Bound): {best_test_acc.mean():.4f} ± {best_test_acc.std():.4f}')


end_time = time()
run_time = end_time-begin_time
print("Running Time:", run_time)