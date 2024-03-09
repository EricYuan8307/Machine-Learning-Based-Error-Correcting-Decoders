import torch
import os
from Decode.NNDecoder import SingleLabelNNDecoder

def Mask40(order, device):
    if order == 1:
        mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 1, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [1, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 2:
        mask = torch.tensor([[0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 1, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [1, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 3:
        mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [1, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 4:
        mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [1, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 5:
        mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [1, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 6:
        mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 1, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0],
                             [1, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 7:
        mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 1, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0],
                             [1, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 8:
        mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 1, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 9:
        mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 1, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 10:
        mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 1, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [1, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    return mask

def Mask42(device):
    mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1],
                         [0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    return mask

def Mask43(order, device):
    if order == 1:
        mask = torch.tensor([[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 2:
        mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 3:
        mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 4:
        mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 5:
        mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 6:
        mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 7:
        mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    return mask

def normalize_abs(data):
    normalized = torch.div(torch.abs(data), torch.sum(torch.abs(data), dim=1).unsqueeze(1))
    return normalized

def modify(origin_size, input_size, threshold, Model_type, neuron_number, origin_model, parameter, origin_model_pth, save_pth, i):
    output_size = torch.pow(torch.tensor(2), origin_size)

    # Assuming you have the model class defined somewhere
    model = origin_model(input_size, neuron_number, output_size)
    model.load_state_dict(torch.load(origin_model_pth))

    for name, param in model.named_parameters():
        if name in(parameter):
            # Apply thresholding to the absoluted normalized weight values
            with (torch.no_grad()):  # Ensure that these operations don't track gradients
                abs_normalized = normalize_abs(param.data)
                param.data = torch.where(abs_normalized < threshold, torch.zeros_like(param.data), param.data)

    # Create the directory if it doesn't exist
    os.makedirs(save_pth, exist_ok=True)
    torch.save(model.state_dict(), f"{save_pth}{Model_type}7_edgedeleted43_order{i}.pth") # exclusive for neuron=7

    # model_modified = torch.load(f'{save_pth}{Model_type}_model_modified_hiddenlayer{neuron_number}_BER0.pth')
    # return model_modified

def modify_mask(origin_size, input_size, model_name, neuron_number, origin_model, parameter, origin_model_pth, save_pth, mask):
    output_size = torch.pow(torch.tensor(2), origin_size)

    # Assuming you have the model class defined somewhere
    model = origin_model(input_size, neuron_number, output_size)
    model.load_state_dict(torch.load(origin_model_pth))

    for name, param in model.named_parameters():
        if name in(parameter):
            # Apply thresholding to the absoluted normalized weight values
            with (torch.no_grad()):  # Ensure that these operations don't track gradients
                param.data = param.data * mask

    # Create the directory if it doesn't exist
    os.makedirs(save_pth, exist_ok=True)
    torch.save(model.state_dict(), f"{save_pth}{model_name}.pth") # exclusive for neuron=7

def loadpara(origin_size, input_size, neuron_number, origin_model, origin_model_pth):
    output_size = torch.pow(torch.tensor(2), origin_size)

    model = torch.load(origin_model_pth)
    print("model parameters:",model)

    # Assuming you have the model class defined somewhere
    model = origin_model(input_size, neuron_number, output_size)
    model.load_state_dict(torch.load(origin_model_pth))

    for name, param in model.named_parameters():
        if name in ("hidden.weight", "output.weight"):
            abs_normalized = normalize_abs(param.data)
            print(f"normalized {name}:", abs_normalized)

# # Model Check
# Model_type = "SLNN"
# origin_size = 4
# input_size = 7
# encoder_type = "Hamming74"
# device = "cpu"
# neuron_number = 7
# origin_model = SingleLabelNNDecoder
# parameter = "hidden.weight"
# edge_delete = 43
# order = 1
# mask = Mask43(order, device) # for edge deleted
# model_name = f"{Model_type}{neuron_number}_edgedeleted{edge_delete}_order{order}_{device}"
#
# origin_model_pth = f"{encoder_type}/Result/Model/{Model_type}_decrease_{parameter}_{device}/{Model_type}_model_hiddenlayer39_BER0.pth"
# # origin_model_pth = "Hamming74/Result/Model/SLNN_edgedeleted42_hidden.weight_cpu/SLNN7_edgedeleted42_cpu.pth" # To modify the 43 edges.
# save_pth = f"{encoder_type}/Result/Model/{Model_type}_edgedeleted{edge_delete}_{parameter}_{device}/"  # exclusive for Neuron=7
#
# model_modified = modify_mask(origin_size, input_size, model_name, neuron_number, origin_model, parameter, origin_model_pth, save_pth, mask)
# print("Model Modify Finished")

# Model Check
Model_type = "SLNN"
origin_size = 4
input_size = 7
output_size = torch.pow(torch.tensor(2), origin_size)
encoder_type = "Hamming74"
device = "cpu"
neuron_number = 7
origin_model = SingleLabelNNDecoder
parameter = "output.weight"
edge_delete = 42

origin_model_pth = f"{encoder_type}/Result/Model/{Model_type}_edgedeleted{edge_delete}_hidden.weight_{device}/{Model_type}7_edgedeleted{edge_delete}_{device}.pth"

# Check original model:
# model_para = loadpara(origin_size, input_size, neuron_number, origin_model, origin_model_pth)

# Model modify:
threshold_hidden_output = [0.1078, 0.1085, 0.1102, 0.1127, 0.1129, 0.1130, 0.1135, 0.1142, 0.1149,
        0.1151, 0.1153, 0.1165, 0.1183, 0.1185, 0.1190, 0.1202, 0.1210, 0.1211,
        0.1218, 0.1238, 0.1240, 0.1242, 0.1247, 0.1255, 0.1261, 0.1266, 0.1269,
        0.1281, 0.1303, 0.1310, 0.1327, 0.1337, 0.1356, 0.1365, 0.1369, 0.1379,
        0.1383, 0.1388, 0.1389, 0.1400, 0.1403, 0.1407, 0.1415, 0.1423, 0.1426,
        0.1427, 0.1435, 0.1435, 0.1439, 0.1444, 0.1448, 0.1450, 0.1451, 0.1452,
        0.1461, 0.1461, 0.1467, 0.1468, 0.1469, 0.1471, 0.1473, 0.1475, 0.1476,
        0.1478, 0.1479, 0.1480, 0.1480, 0.1487, 0.1489, 0.1490, 0.1490, 0.1502,
        0.1502, 0.1505, 0.1509, 0.1510, 0.1515, 0.1522, 0.1523, 0.1529, 0.1533,
        0.1534, 0.1538, 0.1538, 0.1539, 0.1541, 0.1544, 0.1553, 0.1555, 0.1558,
        0.1563, 0.1565, 0.1567, 0.1572, 0.1582, 0.1586, 0.1587, 0.1598, 0.1604,
        0.1606, 0.1611, 0.1619, 0.1637, 0.1669, 0.1674, 0.1678, 0.1696, 0.1711,
        0.1742, 0.1779, 0.1784, 0.1823] # threshold for hidden and output weights



neuron_number_modify = torch.arange(0, 113, 1)
save_pth = f"{encoder_type}/Result/Model/{Model_type}_edgedeleted43_output.weight_{device}/"  # exclusive for Neuron=7
for i in range(len((threshold_hidden_output))):
    model_modified = modify(origin_size, input_size, threshold_hidden_output[i], Model_type, neuron_number, origin_model, parameter, origin_model_pth, save_pth, i)
print("Model Modify Finished")

# # model inspect:
# neuron_number_inspect = 7
# threshold_inspect = 0.1 # normalized
#
# modified_model_pth = f"{encoder_type}/Result/Model/{Model_type}_modified_threshold{threshold_inspect}_{device}/{Model_type}_model_modified_hiddenlayer{neuron_number_inspect}_BER0.pth"
# # print("modified model:", torch.load(modified_model_pth))