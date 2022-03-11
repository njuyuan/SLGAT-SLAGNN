import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, download_url, Data
from sklearn.metrics.pairwise import cosine_similarity as cos


def get_binarized_kneighbors_graph(features, topk, mask=None, device=None):
    assert features.requires_grad is False
    # Compute cosine similarity matrix
    features_norm = features.div(torch.norm(features, p=2, dim=-1, keepdim=True)) 
    attention = torch.matmul(features_norm, features_norm.transpose(-1, -2))

    if mask is not None:
        attention = attention.masked_fill_(1 - mask.byte().unsqueeze(1), 0)
        attention = attention.masked_fill_(1 - mask.byte().unsqueeze(-1), 0)

    # Extract and Binarize kNN-graph
    topk = min(topk, attention.size(-1))
    _, knn_ind = torch.topk(attention, topk, dim=-1) 
    adj = torch.zeros_like(attention).scatter_(-1, knn_ind, 1) 
    # scatter_(input, dim, index, src)
    return adj

def build_knn_neighbourhood(self, attention, topk, markoff_value): # markoff_value= 0 or -INF
    topk = min(topk, attention.size(-1))
    knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
    weighted_adjacency_matrix = to_cuda((markoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val),
                                            self.device)
    return weighted_adjacency_matrix

def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
    mask = (attention > epsilon).detach().float()
    weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
    return weighted_adjacency_matrix


def add_graph_loss(self, out_adj, features):
    # Graph regularization
    graph_loss = 0
    L = torch.diagflat(torch.sum(out_adj, -1)) - out_adj  # L=D-A
    graph_loss += self.config['smoothness_ratio'] * torch.trace(
        torch.mm(features.transpose(-1, -2), torch.mm(L, features))) / int(np.prod(out_adj.shape))
    ones_vec = to_cuda(torch.ones(out_adj.size(-1)), self.device)
    graph_loss += -self.config['degree_ratio'] * torch.mm(ones_vec.unsqueeze(0), torch.log(
        torch.mm(out_adj, ones_vec.unsqueeze(-1)) + Constants.VERY_SMALL_NUMBER)).squeeze() / out_adj.shape[-1]
    graph_loss += self.config['sparsity_ratio'] * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
    return graph_loss

def add_batch_graph_loss(self, out_adj, features, keep_batch_dim=False):
        # Graph regularization
    if keep_batch_dim:
        graph_loss = []
        for i in range(out_adj.shape[0]):
            L = torch.diagflat(torch.sum(out_adj[i], -1)) - out_adj[i]
            graph_loss.append(self.config['smoothness_ratio'] * torch.trace(
                torch.mm(features[i].transpose(-1, -2), torch.mm(L, features[i]))) / int(
                np.prod(out_adj.shape[1:])))

        graph_loss = to_cuda(torch.Tensor(graph_loss), self.device)

        ones_vec = to_cuda(torch.ones(out_adj.shape[:-1]), self.device)
        graph_loss += -self.config['degree_ratio'] * torch.matmul(ones_vec.unsqueeze(1), torch.log(
            torch.matmul(out_adj, ones_vec.unsqueeze(-1)) + Constants.VERY_SMALL_NUMBER)).squeeze(-1).squeeze(-1) / \
                        out_adj.shape[-1]
        graph_loss += self.config['sparsity_ratio'] * torch.sum(torch.pow(out_adj, 2), (1, 2)) / int(
            np.prod(out_adj.shape[1:]))


    else:
        graph_loss = 0
        for i in range(out_adj.shape[0]):
            L = torch.diagflat(torch.sum(out_adj[i], -1)) - out_adj[i]
            graph_loss += self.config['smoothness_ratio'] * torch.trace(
                    torch.mm(features[i].transpose(-1, -2), torch.mm(L, features[i]))) / int(np.prod(out_adj.shape))

        ones_vec = to_cuda(torch.ones(out_adj.shape[:-1]), self.device)
        graph_loss += -self.config['degree_ratio'] * torch.matmul(ones_vec.unsqueeze(1), torch.log(
            torch.matmul(out_adj, ones_vec.unsqueeze(-1)) + Constants.VERY_SMALL_NUMBER)).sum() / out_adj.shape[0] / \
                          out_adj.shape[-1]
        graph_loss += self.config['sparsity_ratio'] * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
    return graph_loss


def get_top_k_matrix(A: np.ndarray, k: int = 10) -> np.ndarray:
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes) 
    A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.  

    norm = A.sum(axis=0)  
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A #/norm  


def get_clipped_matrix(A: np.ndarray, eps: float = 0.1) -> np.ndarray:
    num_nodes = A.shape[0]
    A[A < eps] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A#/norm


def get_adj_matrix(data):
    num_nodes = data.x.shape[0]
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i, j in zip(data.edge_index[0], data.edge_index[1]):
        adj_matrix[i, j] = 1.
    return adj_matrix


def get_feature_matrix(data):
    feature_matrix = cos(data.x)
    return feature_matrix

def normalize_adj(adj_matrix, self_loop=True):
    """Symmetrically normalize adjacency matrix."""
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)  
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return H


def create_data(data, ppmi_matrix):
#def create_data(data):
    adj_matrix = get_adj_matrix(data)
    adj_matrix = normalize_adj(adj_matrix)
 #   print("adj_matrix: ", adj_matrix)

    # A+A^2 
    ppmi_matrix = adj_matrix + ppmi_matrix # adj+ppmi
    ppmi_matrix = normalize_adj(ppmi_matrix)

    edges_i = []
    edges_j = []
    edge_attr = []
    for i, row in enumerate(ppmi_matrix):
        for j in np.where(row > 0)[0]:
            edges_i.append(i)
            edges_j.append(j)
            edge_attr.append(ppmi_matrix[i, j])
    edge_index = [edges_i, edges_j]
    data = Data(
        x=data.x,
        edge_index=torch.LongTensor(edge_index),
        edge_attr=torch.FloatTensor(edge_attr),
        y=data.y,
        train_mask=torch.zeros(data.train_mask.size()[0], dtype=torch.bool),
        test_mask=torch.zeros(data.test_mask.size()[0], dtype=torch.bool),
        val_mask=torch.zeros(data.val_mask.size()[0], dtype=torch.bool)
    )
    return data.edge_index, data.edge_attr
