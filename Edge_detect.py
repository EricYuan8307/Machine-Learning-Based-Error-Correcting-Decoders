import torch

def edge_detection(model_pth):
    # model_pth = f"Result/Model/{encoding_method}{encoded}_{bits}/{NN_type}_{SLNN_hidden_size}_deleted_{device}/{NN_type}_deleted{edge_delete}.pth"
    model = torch.load(model_pth)
    mask = (model['hidden.weight'] != 0).int()
    edge_positions = (mask == 1).nonzero()
    return edge_positions
