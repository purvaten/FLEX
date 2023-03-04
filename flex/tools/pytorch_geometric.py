import scipy.sparse
import torch


def to_scipy_sparse_matrix(
    edge_index,
    num_nodes=None,
) -> scipy.sparse.coo_matrix:

    row, col = edge_index
    edge_attr = torch.ones(row.shape[0])
    out = scipy.sparse.coo_matrix((edge_attr, (row, col)), (num_nodes, num_nodes))
    return out
