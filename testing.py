import torch
import numpy as np

def read_alist(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Read the number of variable and check nodes
    n_var_nodes, n_check_nodes = map(int, lines[0].split())

    # Read the maximum number of edges
    max_var_edges, max_check_edges = map(int, lines[1].split())

    # Read the variable node edges
    var_node_edges = []
    for i in range(2, 2 + n_var_nodes):
        edges = list(map(int, lines[i].split()))
        var_node_edges.append(edges)

    # Read the check node edges
    check_node_edges = []
    for i in range(2 + n_var_nodes, 2 + n_var_nodes + n_check_nodes):
        edges = list(map(int, lines[i].split()))
        check_node_edges.append(edges)

    return n_var_nodes, n_check_nodes, var_node_edges, check_node_edges


def alist_to_parity_check_matrix(alist_path):
    n_var_nodes, n_check_nodes, var_node_edges, check_node_edges = read_alist(alist_path)

    # Initialize the parity check matrix with zeros
    H = torch.zeros((n_check_nodes, n_var_nodes), dtype=torch.int)

    # Populate the parity check matrix
    for cn_index, edges in enumerate(check_node_edges):
        for edge in edges:
            if edge != 0:  # edges are 1-based in .alist, 0 means no edge
                H[cn_index, edge - 1] = 1

    return H


# Example usage:
alist_path = 'Matrix/wifi_648_r083.alist'
H = alist_to_parity_check_matrix(alist_path)

# Display the parity-check matrix
print(H.shape)

def save_parity_check_matrix_as_text(H, file_path):
    np.savetxt(file_path, H.numpy(), fmt='%d')
    print(f"Parity-check matrix saved to {file_path}")

# Example usage:
text_file_path = 'Matrix/parity_check_matrix.txt'
save_parity_check_matrix_as_text(H, text_file_path)

