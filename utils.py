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


def track_trans(adj_matrix, new_adj_matrix, ppr_mat):
    num_nodes = adj_matrix.shape[0]
    e = np.ones((num_nodes,), dtype=int)

    adj_matrix = adj_matrix + np.diag(e)
    deg_matrix = np.diag(np.dot(adj_matrix, e))
    trans_matrix = np.dot(adj_matrix, np.linalg.inv(deg_matrix))

    new_adj_matrix = new_adj_matrix + np.diag(e)
    new_deg_matrix = np.diag(np.dot(new_adj_matrix, e))
    new_trans_matrix = np.dot(new_adj_matrix, np.linalg.inv(new_deg_matrix))

    # --- push to approximate converged --- #
    diff_matrix = new_trans_matrix - trans_matrix
    pushout = 0.85 * np.dot(diff_matrix, np.transpose(ppr_mat))  # - probability mass that needs to be pushed out - #
    cumul_pushout = pushout                   # k = 0

    temp = 0.85 * new_trans_matrix            # k = 1
    cumul_pushout += np.dot(temp, pushout)

    num_itr = 1                               # k starts from 2 to user-specified number of iterations
    for k in range(num_itr):
        new_temp = np.dot(temp * 0.85, new_trans_matrix)
        cumul_pushout += np.dot(new_temp, pushout)
        temp = new_temp

    new_ppr_mat = ppr_mat + np.transpose(cumul_pushout)

    return new_ppr_mat