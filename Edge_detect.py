import torch

def edge_detection(model_pth, weight_place):
    # model_pth = f"Result/Model/{encoding_method}{encoded}_{bits}/{NN_type}_{SLNN_hidden_size}_deleted_{device}/{NN_type}_deleted{edge_delete}.pth"
    model = torch.load(model_pth)
    weight = model[weight_place]
    mask = (weight != 0).int()
    edge_positions = (mask == 1).nonzero()
    return weight, edge_positions

model_pth = "Result/Model/Parity26_10/24_ft_cpu/SLNN_deleted550_trained.pth"
weight_place = "output.weight"
weight, edge_positions = edge_detection(model_pth, weight_place)
# print("weight:", weight)
print("edge_positions:", edge_positions)