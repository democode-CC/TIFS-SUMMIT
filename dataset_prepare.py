import os
import math
import pickle
import torch
import pandas as pd
import networkx as nx
from tqdm import tqdm
from torch_geometric.seed import seed_everything
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import CitationFull, Coauthor
from torch_geometric.utils import train_test_split_edges, k_hop_subgraph, negative_sampling, is_undirected, to_networkx

from utils import *

data_dir = './data'
df_size = [i / 100 for i in range(10)] + [i / 10 for i in range(10)] + [i for i in range(10)]
seeds = [42, 21, 13, 87, 100]
graph_datasets = ['Cora', 'PubMed', 'DBLP', 'CS']
os.makedirs(data_dir, exist_ok=True)




def train_test_split_edges_no_neg_adj_mask(data, val_ratio: float = 0.05, test_ratio: float = 0.1, two_hop_degree=None,
                                           kg=False):
    '''Avoid adding neg_adj_mask'''

    num_nodes = data.num_nodes
    row, col = data.edge_index
    edge_attr = data.edge_attr
    data.edge_index = data.edge_attr = data.edge_weight = data.edge_year = data.edge_type = None


    mask = row < col
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    if two_hop_degree is not None:
        low_degree_mask = two_hop_degree < 50

        low = low_degree_mask.nonzero().squeeze()
        high = (~low_degree_mask).nonzero().squeeze()

        low = low[torch.randperm(low.size(0))]
        high = high[torch.randperm(high.size(0))]

        perm = torch.cat([low, high])

    else:
        perm = torch.randperm(row.size(0))

    row = row[perm]
    col = col[perm]

    # Train
    r, c = row[n_v + n_t:], col[n_v + n_t:]


    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        # out = to_undirected(data.train_pos_edge_index, edge_attr[n_v + n_t:])
        data.train_pos_edge_index, data.train_pos_edge_attr = out
    else:
        data.train_pos_edge_index = data.train_pos_edge_index
        # data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    assert not is_undirected(data.train_pos_edge_index)

    # Test
    r, c = row[:n_t], col[:n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)


    neg_edge_index = negative_sampling(
        edge_index=data.test_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.test_pos_edge_index.shape[1])

    data.test_neg_edge_index = neg_edge_index

    # Valid
    r, c = row[n_t:n_t + n_v], col[n_t:n_t + n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)



    neg_edge_index = negative_sampling(
        edge_index=data.val_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.val_pos_edge_index.shape[1])

    data.val_neg_edge_index = neg_edge_index

    return data


def process_graph():
    for d in graph_datasets:

        if d in ['Cora', 'PubMed', 'DBLP']:
            dataset = CitationFull(os.path.join(data_dir, d), d, transform=T.NormalizeFeatures())
        elif d in ['CS']:
            dataset = Coauthor(os.path.join(data_dir, d), d, transform=T.NormalizeFeatures())
        else:
            raise NotImplementedError

        print('Processing:', d)
        print(dataset)
        data = dataset[0]
        data.train_mask = data.val_mask = data.test_mask = None
        graph = to_networkx(data)

        # Get two hop degree for all nodes
        node_to_neighbors = {}
        for n in tqdm(graph.nodes(), desc='Two hop neighbors'):
            neighbor_1 = set(graph.neighbors(n))
            neighbor_2 = sum([list(graph.neighbors(i)) for i in neighbor_1], [])
            neighbor_2 = set(neighbor_2)
            neighbor = neighbor_1 | neighbor_2

            node_to_neighbors[n] = neighbor

        two_hop_degree = []
        row, col = data.edge_index
        mask = row < col
        row, col = row[mask], col[mask]
        for r, c in tqdm(zip(row, col), total=len(row)):
            neighbor_row = node_to_neighbors[r.item()]
            neighbor_col = node_to_neighbors[c.item()]
            neighbor = neighbor_row | neighbor_col

            num = len(neighbor)

            two_hop_degree.append(num)

        two_hop_degree = torch.tensor(two_hop_degree)

        for s in seeds:
            seed_everything(s)

            # D
            data = dataset[0]

            data = train_test_split_edges_no_neg_adj_mask(data, test_ratio=0.05)
            print(s, data)

            with open(os.path.join(data_dir, d, f'd_{s}.pkl'), 'wb') as f:
                pickle.dump((dataset, data), f)
            _, local_edges, _, mask = k_hop_subgraph(
                data.test_pos_edge_index.flatten().unique(),
                2,
                data.train_pos_edge_index,
                num_nodes=dataset[0].num_nodes)
            distant_edges = data.train_pos_edge_index[:, ~mask]
            print('Number of edges. Local: ', local_edges.shape[1], 'Distant:', distant_edges.shape[1])

            in_mask = mask
            out_mask = ~mask
            torch.save(
                {'out': out_mask, 'in': in_mask},
                os.path.join(data_dir, d, f'df_{s}.pt')
            )





def main():
    process_graph()


if __name__ == "__main__":
    main()
