import numpy as np
from collections import defaultdict
import torch

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # 路径压缩
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

def find_connected_components(similarity_matrix):
    n = similarity_matrix.shape[0]
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n): 
            if similarity_matrix[i, j]:
                uf.union(i, j)

    merged_labels = defaultdict(list)
    for i in range(n):
        root = uf.find(i)
        merged_labels[root].append(i)

    connected_components = []
    for component in merged_labels.values():
        if len(component) == 1:
            connected_components.append(component[0]) 
        else:
            connected_components.append(list(component))  

    return connected_components


def update_super_index(subgraph_node_id, num_nodes):
    super_index = torch.zeros(num_nodes, dtype=torch.long)
    
    for sp_id, nodes in enumerate(subgraph_node_id):
        super_index[nodes] = sp_id
    
    return super_index