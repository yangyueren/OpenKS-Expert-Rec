import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

def get_edge_index_except_these_nodes(nodeids, edge_index, dim):
    """

    Args:
        nodeids (torch.tensor): (node_num,)
        edge_index (torch.tensor): (2, num_edges)
        dim (int): 0 or 1
    """
    for node in nodeids:
        mask = (edge_index[dim] == node)
        edge_index[dim][mask] = -1
    mask = edge_index[dim] != -1
    ans_edge_index = edge_index[:, mask]
    return ans_edge_index


def sampler_of_graph(nodeids, node_name, G, neighbor_num=15, iterations=3):
    # import pdb; pdb.set_trace()
    nodeidl = list(set(nodeids.cpu().tolist()))
    mask = torch.zeros(len(G[node_name].x))
    mask[nodeidl] = 1.0
    mask = mask.bool().to(G[node_name].x).bool()
    loader = NeighborLoader(
        G,
        # Sample 30 neighbors for each node and edge type for 2 iterations
        num_neighbors={key: [neighbor_num] * iterations for key in G.edge_types},
        # Use a batch size of 128 for sampling training nodes of type paper
        batch_size=len(nodeidl),
        # batch_size=4,
        input_nodes=(node_name, mask),
    )

    sampled_graph = next(iter(loader))
    # nids = set(sampled_graph[node_name].n_id.tolist())
    # for i in nodeidl:
    #     assert i in nids

    return sampled_graph

