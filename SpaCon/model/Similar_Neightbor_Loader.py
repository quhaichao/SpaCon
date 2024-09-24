import torch
import numpy as np
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.sampler import BaseSampler



class SimilarNeighborSampler(BaseSampler):
    def __init__(self, data, sizes, correlation_matrix, k_neighbors):
        super().__init__()
        self.data = data
        self.sizes = sizes
        self.correlation_matrix = correlation_matrix
        self.k_neighbors = k_neighbors
        # 构建邻接列表
        self.adj_dict = {i: set() for i in range(data.num_nodes)}
        edge_index = data.edge_index
        for src, dst in edge_index.t().tolist():
            self.adj_dict[src].add(dst)
            self.adj_dict[dst].add(src)
    
    def sample_from_nodes(self, batch):
        n_id = batch.tolist()
        e_id = []
        sampled_nodes = set(n_id)
        node_batches = [n_id]

        for layer, (size, k) in enumerate(zip(self.sizes, self.k_neighbors)):
            num_random = size - k
            new_n_id = []
            for node in node_batches[-1]:
                neighbors = list(self.adj_dict[node])
                if not neighbors:
                    continue
                # 获取相关性
                correlations = self.correlation_matrix[node, neighbors]
                sorted_indices = np.argsort(-correlations)
                top_k = [neighbors[idx] for idx in sorted_indices[:k]]
                remaining_neighbors = list(set(neighbors) - set(top_k))
                if remaining_neighbors and num_random > 0:
                    random_neighbors = np.random.choice(
                        remaining_neighbors,
                        size=min(num_random, len(remaining_neighbors)),
                        replace=False
                    ).tolist()
                else:
                    random_neighbors = []
                sampled_neighbors = top_k + random_neighbors
                new_n_id.extend(sampled_neighbors)
                e_id.extend([(node, nbr) for nbr in sampled_neighbors])
            node_batches.append(new_n_id)
            sampled_nodes.update(new_n_id)
        
        # 构建子图
        subgraph_nodes = list(sampled_nodes)
        node_id_map = {node_id: idx for idx, node_id in enumerate(subgraph_nodes)}
        edge_index = torch.tensor(
            [[node_id_map[src], node_id_map[dst]] for src, dst in e_id],
            dtype=torch.long
        ).t().contiguous()
        # 创建子图数据对象
        sub_data = Data(
            x=self.data.x[subgraph_nodes],
            edge_index=edge_index,
            y=self.data.y[subgraph_nodes] if self.data.y is not None else None,
            n_id=torch.tensor(subgraph_nodes, dtype=torch.long)
        )
        return sub_data



def similar_neighbor_loder(data, correlation_matrix, batch_size=32, num_neighbors=[20, 10, 10], beta=0.1):
    k_similar_neighbors = [n*beta for n in num_neighbors]
    sampler = SimilarNeighborSampler(data, num_neighbors, correlation_matrix, k_similar_neighbors)
    node_dataset = torch.arange(data.num_nodes)

    def collate_fn(batch):
        batch = torch.tensor(batch, dtype=torch.long)
        sub_data = sampler.sample_from_nodes(batch)
        return sub_data

    loader = DataLoader(
        node_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    return loader