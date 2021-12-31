import numpy as np

def power_iteration(ppr, adj_matrix):
    res = ppr
    num_nodes = adj_matrix.shape[0]
    e = np.ones((num_nodes,), dtype=int)
    adj_matrix = adj_matrix + np.diag(e)
    deg_matrix = np.diag(np.dot(adj_matrix, e))
    for i in range(20):
        ppr = 0.85 * np.dot(np.dot(adj_matrix, np.linalg.inv(deg_matrix)), ppr) + 0.15 * res
    return ppr