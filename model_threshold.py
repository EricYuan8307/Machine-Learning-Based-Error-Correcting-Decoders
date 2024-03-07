import torch
import os
from Decode.NNDecoder import SingleLabelNNDecoder

def Mask(order, device):
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

def normalize_abs(data):
    normalized = torch.div(torch.abs(data), torch.sum(torch.abs(data), dim=1).unsqueeze(1))
    return normalized

def modify(origin_size, input_size, threshold, Model_type, neuron_number, origin_model, parameter, origin_model_pth, save_pth, edge_delete):
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
    torch.save(model.state_dict(), f"{save_pth}{Model_type}_model_deleted{edge_delete}_BER0.pth") # exclusive for neuron=7

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

# Model Check
Model_type = "SLNN"
origin_size = 4
input_size = 7
encoder_type = "Hamming74"
device = "cpu"
neuron_number = 7
origin_model = SingleLabelNNDecoder
parameter = "hidden.weight"
edge_delete = 39
order = 10
model_name = f"{Model_type}{neuron_number}_edgedeleted40_order{order}_{device}"
mask = Mask(order, device)

origin_model_pth = f"{encoder_type}/Result/Model/{Model_type}_decrease_{parameter}_{device}/{Model_type}_model_hiddenlayer39_BER0.pth"

neuron_number_modify = 14
save_pth = f"{encoder_type}/Result/Model/{Model_type}_edgedeleted40_{parameter}_{device}/"  # exclusive for Neuron=7

model_modified = modify_mask(origin_size, input_size, model_name, neuron_number, origin_model, parameter, origin_model_pth, save_pth, mask)
print("Model Modify Finished")

# # Model Check
# Model_type = "SLNN"
# origin_size = 4
# input_size = 7
# output_size = torch.pow(torch.tensor(2), origin_size)
# encoder_type = "Hamming74"
# device = "cpu"
# neuron_number = 7
# origin_model = SingleLabelNNDecoder
# parameter = "hidden.weight"
# edge_delete = 9
#
# origin_model_pth = f"{encoder_type}/Result/Model/{Model_type}_decrease_con_{device}_{parameter}/{Model_type}_model_deleted{edge_delete}_BER0.pth"
#
# # Check original model:
# model_para = loadpara(origin_size, input_size, neuron_number, origin_model, origin_model_pth)
#
# # # Model modify:
# # threshold = 0.05 # normalized
# threshold_hidden = [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#                     0.0000e+00, 0.0000e+00, 0.0000e+00, 2.1629e-04, 7.8333e-03, 1.8974e-02,
#                     2.0154e-02, 2.1156e-02, 2.5344e-02, 2.6341e-02, 3.3531e-02, 3.3845e-02,
#                     3.4220e-02, 3.9235e-02, 4.2089e-02, 4.2153e-02, 5.5994e-02, 5.7328e-02,
#                     1.5423e-01, 1.6127e-01, 1.7117e-01, 1.8220e-01, 1.9343e-01, 1.9478e-01,
#                     2.0990e-01, 2.1766e-01, 2.2345e-01, 2.2347e-01, 2.3885e-01, 2.3975e-01,
#                     2.4509e-01, 2.4717e-01, 2.4962e-01, 2.5171e-01, 2.5360e-01, 2.5416e-01,
#                     2.6077e-01, 2.7130e-01, 2.7892e-01, 2.9003e-01, 2.9127e-01, 3.7997e-01,
#                     8.5781e-01] # Proportion for hidden weight
#
# # threshold_output = [0.0005, 0.0008, 0.0034, 0.0167, 0.0174, 0.0194, 0.0415, 0.0474, 0.0501,
# #         0.0502, 0.0533, 0.0643, 0.0695, 0.0896, 0.0908, 0.0936, 0.0947, 0.0982,
# #         0.0989, 0.1019, 0.1092, 0.1136, 0.1149, 0.1179, 0.1209, 0.1230, 0.1236,
# #         0.1255, 0.1286, 0.1299, 0.1303, 0.1304, 0.1320, 0.1325, 0.1335, 0.1342,
# #         0.1342, 0.1346, 0.1377, 0.1385, 0.1405, 0.1407, 0.1407, 0.1411, 0.1415,
# #         0.1415, 0.1419, 0.1430, 0.1431, 0.1446, 0.1451, 0.1453, 0.1460, 0.1482,
# #         0.1486, 0.1501, 0.1502, 0.1509, 0.1514, 0.1518, 0.1520, 0.1559, 0.1559,
# #         0.1560, 0.1585, 0.1588, 0.1594, 0.1610, 0.1624, 0.1626, 0.1634, 0.1638,
# #         0.1646, 0.1655, 0.1656, 0.1659, 0.1665, 0.1681, 0.1684, 0.1684, 0.1687,
# #         0.1705, 0.1708, 0.1715, 0.1718, 0.1731, 0.1731, 0.1733, 0.1749, 0.1750,
# #         0.1786, 0.1799, 0.1824, 0.1841, 0.1860, 0.1871, 0.1883, 0.1891, 0.1925,
# #         0.1929, 0.1938, 0.1946, 0.1947, 0.1969, 0.2056, 0.2064, 0.2126, 0.2265,
# #         0.2341, 0.2396, 0.2453, 0.2698] # Proportion for output weight
# #
# # threshold_hidden_output = [
# #     5.0000e-04, 8.0000e-04, 3.4000e-03, 1.1000e-02, 1.2200e-02, 1.6000e-02,
# #     1.6300e-02, 1.6700e-02, 1.7400e-02, 1.8000e-02, 1.9400e-02, 2.1700e-02,
# #     2.3200e-02, 3.2400e-02, 3.2900e-02, 3.3600e-02, 3.5600e-02, 3.7200e-02,
# #     3.7700e-02, 4.0800e-02, 4.1500e-02, 4.5600e-02, 4.7400e-02, 5.0100e-02,
# #     5.0200e-02, 5.1700e-02, 5.3300e-02, 5.5000e-02, 6.0600e-02, 6.4300e-02,
# #     6.5300e-02, 6.9500e-02, 7.2500e-02, 7.5500e-02, 7.6200e-02, 8.9600e-02,
# #     9.0800e-02, 9.0900e-02, 9.3600e-02, 9.4700e-02, 9.8200e-02, 9.8900e-02,
# #     1.0190e-01, 1.0270e-01, 1.0920e-01, 1.1360e-01, 1.1360e-01, 1.1490e-01,
# #     1.1790e-01, 1.2090e-01, 1.2300e-01, 1.2360e-01, 1.2550e-01, 1.2860e-01,
# #     1.2990e-01, 1.3030e-01, 1.3040e-01, 1.3200e-01, 1.3210e-01, 1.3250e-01,
# #     1.3350e-01, 1.3420e-01, 1.3420e-01, 1.3450e-01, 1.3460e-01, 1.3690e-01,
# #     1.3770e-01, 1.3850e-01, 1.4050e-01, 1.4070e-01, 1.4070e-01, 1.4110e-01,
# #     1.4150e-01, 1.4150e-01, 1.4190e-01, 1.4300e-01, 1.4310e-01, 1.4460e-01,
# #     1.4510e-01, 1.4530e-01, 1.4600e-01, 1.4820e-01, 1.4850e-01, 1.4860e-01,
# #     1.5010e-01, 1.5020e-01, 1.5090e-01, 1.5140e-01, 1.5180e-01, 1.5200e-01,
# #     1.5590e-01, 1.5590e-01, 1.5600e-01, 1.5790e-01, 1.5850e-01, 1.5880e-01,
# #     1.5940e-01, 1.6100e-01, 1.6240e-01, 1.6260e-01, 1.6340e-01, 1.6380e-01,
# #     1.6460e-01, 1.6550e-01, 1.6560e-01, 1.6590e-01, 1.6650e-01, 1.6810e-01,
# #     1.6840e-01, 1.6840e-01, 1.6870e-01, 1.7050e-01, 1.7080e-01, 1.7150e-01,
# #     1.7180e-01, 1.7310e-01, 1.7310e-01, 1.7330e-01, 1.7490e-01, 1.7500e-01,
# #     1.7730e-01, 1.7860e-01, 1.7990e-01, 1.8010e-01, 1.8240e-01, 1.8410e-01,
# #     1.8600e-01, 1.8710e-01, 1.8830e-01, 1.8910e-01, 1.8930e-01, 1.8970e-01,
# #     1.9250e-01, 1.9290e-01, 1.9380e-01, 1.9460e-01, 1.9470e-01, 1.9690e-01,
# #     2.0130e-01, 2.0560e-01, 2.0570e-01, 2.0640e-01, 2.0900e-01, 2.1260e-01,
# #     2.1810e-01, 2.2650e-01, 2.2820e-01, 2.2930e-01, 2.3410e-01, 2.3960e-01,
# #     2.4400e-01, 2.4510e-01, 2.4530e-01, 2.4790e-01, 2.4940e-01, 2.6980e-01,
# #     2.7680e-01, 2.9490e-01, 3.7840e-01, 4.7520e-01, 6.7240e-01] # threshold for hidden and output weights
#
#
# # parameter = "hidden.weight", "output.weight"
# # neuron_number_modify = torch.arange(0, 101, 1)
# neuron_number_modify = 14
# save_pth = f"{encoder_type}/Result/Model/{Model_type}_decrease_con_{device}_{parameter}/"  # exclusive for Neuron=7
# #
# #
#
# model_modified = modify(origin_size, input_size, threshold_hidden[neuron_number_modify], Model_type, neuron_number, origin_model, parameter, origin_model_pth, save_pth, neuron_number_modify)
# print("Model Modify Finished")
#
# # model inspect:
# neuron_number_inspect = 7
# threshold_inspect = 0.1 # normalized
#
# # modified_model_pth = f"{encoder_type}/Result/Model/{Model_type}_modified_threshold{threshold_inspect}_{device}/{Model_type}_model_modified_hiddenlayer{neuron_number_inspect}_BER0.pth"
# # print("modified model:", torch.load(modified_model_pth))