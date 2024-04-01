import torch
import os
from Decode.NNDecoder import SingleLabelNNDecoder1


# def Mask40(order, device):
#     if order == 1:
#         mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
#                              [0, 0, 0, 1, 0, 1, 0],
#                              [0, 0, 0, 0, 1, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 1],
#                              [0, 1, 0, 0, 0, 0, 0],
#                              [1, 0, 1, 0, 0, 0, 0],
#                              [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)
#
#     if order == 2:
#         mask = torch.tensor([[0, 0, 1, 0, 0, 0, 0],
#                              [0, 0, 0, 1, 0, 1, 0],
#                              [0, 0, 0, 0, 1, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 1],
#                              [0, 1, 0, 0, 0, 0, 0],
#                              [1, 0, 1, 0, 0, 0, 0],
#                              [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)
#
#     if order == 3:
#         mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
#                              [0, 0, 0, 0, 0, 1, 0],
#                              [0, 0, 0, 0, 1, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 1],
#                              [0, 1, 0, 0, 0, 0, 0],
#                              [1, 0, 1, 0, 0, 0, 0],
#                              [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)
#
#     if order == 4:
#         mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
#                              [0, 0, 0, 1, 0, 0, 0],
#                              [0, 0, 0, 0, 1, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 1],
#                              [0, 1, 0, 0, 0, 0, 0],
#                              [1, 0, 1, 0, 0, 0, 0],
#                              [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)
#
#     if order == 5:
#         mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
#                              [0, 0, 0, 1, 0, 1, 0],
#                              [0, 0, 0, 0, 0, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 1],
#                              [0, 1, 0, 0, 0, 0, 0],
#                              [1, 0, 1, 0, 0, 0, 0],
#                              [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)
#
#     if order == 6:
#         mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
#                              [0, 0, 0, 1, 0, 1, 0],
#                              [0, 0, 0, 0, 1, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 0],
#                              [0, 1, 0, 0, 0, 0, 0],
#                              [1, 0, 1, 0, 0, 0, 0],
#                              [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)
#
#     if order == 7:
#         mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
#                              [0, 0, 0, 1, 0, 1, 0],
#                              [0, 0, 0, 0, 1, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 1],
#                              [0, 0, 0, 0, 0, 0, 0],
#                              [1, 0, 1, 0, 0, 0, 0],
#                              [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)
#
#     if order == 8:
#         mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
#                              [0, 0, 0, 1, 0, 1, 0],
#                              [0, 0, 0, 0, 1, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 1],
#                              [0, 1, 0, 0, 0, 0, 0],
#                              [0, 0, 1, 0, 0, 0, 0],
#                              [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)
#
#     if order == 9:
#         mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
#                              [0, 0, 0, 1, 0, 1, 0],
#                              [0, 0, 0, 0, 1, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 1],
#                              [0, 1, 0, 0, 0, 0, 0],
#                              [1, 0, 0, 0, 0, 0, 0],
#                              [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)
#
#     if order == 10:
#         mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
#                              [0, 0, 0, 1, 0, 1, 0],
#                              [0, 0, 0, 0, 1, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 1],
#                              [0, 1, 0, 0, 0, 0, 0],
#                              [1, 0, 1, 0, 0, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)
#
#     return mask
#
# def Mask42(device):
#     mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
#                          [0, 0, 0, 1, 0, 0, 0],
#                          [0, 0, 0, 0, 1, 0, 0],
#                          [0, 0, 0, 0, 0, 0, 1],
#                          [0, 1, 0, 0, 0, 0, 0],
#                          [0, 0, 1, 0, 0, 0, 0],
#                          [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)
#
#     return mask
#
# def Mask43(order, device):
#     if order == 1:
#         mask = torch.tensor([[0, 0, 0, 0, 0, 0, 0],
#                              [0, 0, 0, 1, 0, 0, 0],
#                              [0, 0, 0, 0, 1, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 1],
#                              [0, 1, 0, 0, 0, 0, 0],
#                              [0, 0, 1, 0, 0, 0, 0],
#                              [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)
#
#     if order == 2:
#         mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
#                              [0, 0, 0, 0, 0, 0, 0],
#                              [0, 0, 0, 0, 1, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 1],
#                              [0, 1, 0, 0, 0, 0, 0],
#                              [0, 0, 1, 0, 0, 0, 0],
#                              [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)
#
#     if order == 3:
#         mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
#                              [0, 0, 0, 1, 0, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 1],
#                              [0, 1, 0, 0, 0, 0, 0],
#                              [0, 0, 1, 0, 0, 0, 0],
#                              [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)
#
#     if order == 4:
#         mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
#                              [0, 0, 0, 1, 0, 0, 0],
#                              [0, 0, 0, 0, 1, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 0],
#                              [0, 1, 0, 0, 0, 0, 0],
#                              [0, 0, 1, 0, 0, 0, 0],
#                              [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)
#
#     if order == 5:
#         mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
#                              [0, 0, 0, 1, 0, 0, 0],
#                              [0, 0, 0, 0, 1, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 1],
#                              [0, 0, 0, 0, 0, 0, 0],
#                              [0, 0, 1, 0, 0, 0, 0],
#                              [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)
#
#     if order == 6:
#         mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
#                              [0, 0, 0, 1, 0, 0, 0],
#                              [0, 0, 0, 0, 1, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 1],
#                              [0, 1, 0, 0, 0, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 0],
#                              [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)
#
#     if order == 7:
#         mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
#                              [0, 0, 0, 1, 0, 0, 0],
#                              [0, 0, 0, 0, 1, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 1],
#                              [0, 1, 0, 0, 0, 0, 0],
#                              [0, 0, 1, 0, 0, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)
#
#     return mask

def normalize_abs(data):
    normalized = torch.div(torch.abs(data), torch.sum(torch.abs(data), dim=1).unsqueeze(1))
    return normalized


def modify(origin_size, input_size, threshold, neuron_number, origin_model, parameter, origin_model_pth, save_pth):
    output_size = torch.pow(torch.tensor(2),
                            origin_size)  # Filter out the model absoluted normalized parameter that is smaller than the threshold

    # Assuming you have the model class defined somewhere
    model = origin_model(input_size, neuron_number, output_size)
    model.load_state_dict(torch.load(origin_model_pth))

    for name, param in model.named_parameters():
        if name in (parameter):
            # Apply thresholding to the absoluted normalized weight values
            with (torch.no_grad()):  # Ensure that these operations don't track gradients
                abs_normalized = normalize_abs(param.data)
                param.data = torch.where(abs_normalized < threshold, torch.zeros_like(param.data), param.data)
                mask = (param.data != 0).int()

                num_zeros = (mask == 0).sum().item()
                model_name = f"{Model_type}_deleted{num_zeros}"

    # Create the directory if it doesn't exist
    os.makedirs(save_pth, exist_ok=True)
    torch.save(model.state_dict(), f"{save_pth}{model_name}.pth")

    # model_modified = torch.load(f'{save_pth}{Model_type}_model_modified_hiddenlayer{neuron_number}_BER0.pth')
    # return model_modified


def modify_exact(origin_size, input_size, position, neuron_number, origin_model, parameter, origin_model_pth, save_pth,
                 order):
    output_size = torch.pow(torch.tensor(2),
                            origin_size)  # Filter out the model parameter that is exactly same as the threshold

    # Assuming you have the model class defined somewhere
    model = origin_model(input_size, neuron_number, output_size)
    model.load_state_dict(torch.load(origin_model_pth))

    for name, param in model.named_parameters():
        if name in (parameter):
            # Apply thresholding to the absoluted normalized weight values
            with (torch.no_grad()):  # Ensure that these operations don't track gradients
                mask = (param.data != 0).int()
                mask[position[0], position[1]] = 0
                param.data = param.data * mask
                num_zeros = (param.data == 0).sum().item()

    model_name = f"{Model_type}_deleted{num_zeros}_order{order}"

    # Create the directory if it doesn't exist
    os.makedirs(save_pth, exist_ok=True)
    torch.save(model.state_dict(), f"{save_pth}{model_name}.pth")

def modify_exact_all(origin_size, input_size, positions, neuron_number, origin_model, parameter, origin_model_pth, save_pth,
                 order):
    output_size = torch.pow(torch.tensor(2),
                            origin_size)  # Filter out the model parameter that is exactly same as the threshold

    # Assuming you have the model class defined somewhere
    model = origin_model(input_size, neuron_number, output_size)
    model.load_state_dict(torch.load(origin_model_pth))

    # To delete all edge in one time
    for name, param in model.named_parameters():
        if name in (parameter):
            # Apply thresholding to the absoluted normalized weight values
            with (torch.no_grad()):  # Ensure that these operations don't track gradients
                for position in positions:
                    mask = (param.data != 0).int()
                    mask[position[0], position[1]] = 0
                    param.data = param.data * mask
                    num_zeros = (param.data == 0).sum().item()

    model_name = f"{Model_type}_deleted{num_zeros}_order{order}"

    # Create the directory if it doesn't exist
    os.makedirs(save_pth, exist_ok=True)
    torch.save(model.state_dict(), f"{save_pth}{model_name}.pth")

def modify_mask(origin_size, input_size, model_name, neuron_number, origin_model, parameter, origin_model_pth, save_pth,
                mask):
    output_size = torch.pow(torch.tensor(2), origin_size)

    # Assuming you have the model class defined somewhere
    model = origin_model(input_size, neuron_number, output_size)
    model.load_state_dict(torch.load(origin_model_pth))

    for name, param in model.named_parameters():
        if name in (parameter):
            # Apply thresholding to the absoluted normalized weight values
            with (torch.no_grad()):  # Ensure that these operations don't track gradients
                param.data = param.data * mask

    # Create the directory if it doesn't exist
    os.makedirs(save_pth, exist_ok=True)
    torch.save(model.state_dict(), f"{save_pth}{model_name}.pth")  # exclusive for neuron=7


def loadpara(origin_size, input_size, neuron_number, origin_model, origin_model_pth):
    output_size = torch.pow(torch.tensor(2), origin_size)

    model = torch.load(origin_model_pth)
    print("model parameters:", model)

    # Assuming you have the model class defined somewhere
    model = origin_model(input_size, neuron_number, output_size)
    model.load_state_dict(torch.load(origin_model_pth))

    for name, param in model.named_parameters():
        if name in ("hidden.weight", "output.weight"):
            abs_normalized = normalize_abs(param.data)
            print(f"normalized {name}:", abs_normalized)


# Model_type = "SLNN"
# bits = 10
# input_size = 26
# output_size = torch.pow(torch.tensor(2), bits)
# encoder_type = "Parity"
# device = "cpu"
# neuron_number = 26
# origin_model = SingleLabelNNDecoder1
# parameter = "hidden.weight"
# # edge_delete = 42
#
# fc_model_pth = f"Result/Model/{encoder_type}{input_size}_{bits}/{Model_type}_{device}/{Model_type}_hiddenlayer{neuron_number}.pth"
# nonfc_model_pth = f"{encoder_type}/Result/Model/{Model_type}_edgedeleted{edge_delete}_hidden.weight_{device}/{Model_type}7_edgedeleted{edge_delete}_{device}.pth"

# Check original model:
# model_para = loadpara(origin_size, input_size, neuron_number, origin_model, origin_model_pth)

# # Model modify:
# absolute_normalized_hidden_output = [1.6138e-05, 1.5792e-04, 3.0335e-04, 3.6933e-04, 4.6497e-04, 5.5660e-04,
#         6.3351e-04, 8.5733e-04, 8.9654e-04, 1.0302e-03, 1.0819e-03, 1.3221e-03,
#         1.4028e-03, 1.5056e-03, 1.5250e-03, 1.5611e-03, 1.5764e-03, 1.6537e-03,
#         1.7870e-03, 1.9085e-03, 2.0125e-03, 2.0189e-03, 2.0801e-03, 2.2724e-03,
#         2.2940e-03, 2.3936e-03, 2.4136e-03, 2.4943e-03, 2.5377e-03, 2.5876e-03,
#         2.6594e-03, 2.9243e-03, 3.0083e-03, 3.1344e-03, 3.1616e-03, 3.2703e-03,
#         3.3595e-03, 3.4252e-03, 3.5727e-03, 3.9603e-03, 4.0885e-03, 4.1934e-03,
#         4.2484e-03, 4.3359e-03, 4.5488e-03, 4.5591e-03, 4.6877e-03, 4.8010e-03,
#         4.8205e-03, 5.0227e-03, 5.2788e-03, 5.3599e-03, 5.4862e-03, 5.6214e-03,
#         5.6628e-03, 5.8434e-03, 5.9073e-03, 5.9426e-03, 6.0675e-03, 6.2580e-03,
#         6.3467e-03, 6.3642e-03, 6.4482e-03, 6.4902e-03, 6.5333e-03, 6.7405e-03,
#         6.7562e-03, 6.8425e-03, 6.8973e-03, 7.0994e-03, 7.1283e-03, 7.3268e-03,
#         7.3746e-03, 7.6306e-03, 7.7753e-03, 7.8265e-03, 7.9060e-03, 7.9069e-03,
#         8.2441e-03, 8.2564e-03, 8.3804e-03, 8.4030e-03, 8.5502e-03, 8.5851e-03,
#         8.6390e-03, 8.7909e-03, 8.8311e-03, 9.2095e-03, 9.2112e-03, 9.2323e-03,
#         9.2723e-03, 9.2980e-03, 9.4341e-03, 9.4770e-03, 9.6910e-03, 9.7975e-03,
#         9.9311e-03, 1.0012e-02, 1.0208e-02, 1.0252e-02, 1.0314e-02, 1.0368e-02,
#         1.0395e-02, 1.0611e-02, 1.0821e-02, 1.0922e-02, 1.1033e-02, 1.1173e-02,
#         1.1255e-02, 1.1300e-02, 1.1444e-02, 1.1468e-02, 1.1623e-02, 1.2032e-02,
#         1.2058e-02, 1.2391e-02, 1.2415e-02, 1.2584e-02, 1.2603e-02, 1.2656e-02,
#         1.2696e-02, 1.2732e-02, 1.2829e-02, 1.2832e-02, 1.2935e-02, 1.3155e-02,
#         1.3436e-02, 1.3463e-02, 1.3478e-02, 1.3480e-02, 1.3537e-02, 1.3802e-02,
#         1.3833e-02, 1.3931e-02, 1.4052e-02, 1.4052e-02, 1.4132e-02, 1.4226e-02, # 1.4226e-02 duplicated
#         1.4344e-02, 1.4469e-02, 1.4590e-02, 1.4629e-02, 1.4789e-02, 1.5047e-02,
#         1.5065e-02, 1.5088e-02, 1.5155e-02, 1.5166e-02, 1.5488e-02, 1.5628e-02,
#         1.5694e-02, 1.5775e-02, 1.5789e-02, 1.5797e-02, 1.5814e-02, 1.5851e-02,
#         1.5863e-02, 1.5917e-02, 1.6168e-02, 1.6226e-02, 1.6408e-02, 1.6521e-02,
#         1.6592e-02, 1.6602e-02, 1.6638e-02, 1.6788e-02, 1.6858e-02, 1.6983e-02,
#         1.6999e-02, 1.7026e-02, 1.7037e-02, 1.7103e-02, 1.7105e-02, 1.7145e-02,
#         1.7177e-02, 1.7189e-02, 1.7254e-02, 1.7313e-02, 1.7380e-02, 1.7391e-02,
#         1.7402e-02, 1.7409e-02, 1.7645e-02, 1.7700e-02, 1.7992e-02, 1.8148e-02,
#         1.8220e-02, 1.8226e-02, 1.8462e-02, 1.8477e-02, 1.8543e-02, 1.8622e-02,
#         1.8695e-02, 1.8722e-02, 1.8785e-02, 1.8809e-02, 1.8841e-02, 1.8853e-02,
#         1.8862e-02, 1.8997e-02, 1.9021e-02, 1.9042e-02, 1.9201e-02, 1.9201e-02, # 1.9201e-02 duplicated
#         1.9206e-02, 1.9297e-02, 1.9497e-02, 1.9506e-02, 1.9671e-02, 1.9777e-02,
#         1.9985e-02, 1.9998e-02, 2.0059e-02, 2.0066e-02, 2.0086e-02, 2.0107e-02,
#         2.0159e-02, 2.0173e-02, 2.0358e-02, 2.0365e-02, 2.0607e-02, 2.0744e-02,
#         2.0884e-02, 2.0900e-02, 2.1000e-02, 2.1076e-02, 2.1197e-02, 2.1239e-02,
#         2.1346e-02, 2.1474e-02, 2.1681e-02, 2.1693e-02, 2.1705e-02, 2.1730e-02,
#         2.2032e-02, 2.2178e-02, 2.2187e-02, 2.2377e-02, 2.2409e-02, 2.2445e-02,
#         2.2503e-02, 2.2540e-02, 2.2561e-02, 2.2566e-02, 2.2652e-02, 2.2667e-02,
#         2.2701e-02, 2.2720e-02, 2.3036e-02, 2.3055e-02, 2.3081e-02, 2.3202e-02,
#         2.3308e-02, 2.3350e-02, 2.3601e-02, 2.3689e-02, 2.3696e-02, 2.3741e-02,
#         2.3862e-02, 2.3964e-02, 2.4129e-02, 2.4149e-02, 2.4160e-02, 2.4171e-02,
#         2.4524e-02, 2.4571e-02, 2.4652e-02, 2.4711e-02, 2.4757e-02, 2.4997e-02,
#         2.5010e-02, 2.5302e-02, 2.5366e-02, 2.5678e-02, 2.5700e-02, 2.5732e-02,
#         2.5738e-02, 2.5811e-02, 2.5911e-02, 2.5916e-02, 2.6133e-02, 2.6244e-02,
#         2.6356e-02, 2.6681e-02, 2.6848e-02, 2.6873e-02, 2.6895e-02, 2.6925e-02,
#         2.6938e-02, 2.7080e-02, 2.7349e-02, 2.7617e-02, 2.7763e-02, 2.7782e-02, # 2.7782e-02 duplicated
#         2.7782e-02, 2.7866e-02, 2.7960e-02, 2.8186e-02, 2.8282e-02, 2.8309e-02,
#         2.8329e-02, 2.8462e-02, 2.8473e-02, 2.8529e-02, 2.8881e-02, 2.8923e-02,
#         2.8985e-02, 2.9114e-02, 2.9144e-02, 2.9713e-02, 2.9756e-02, 3.0124e-02,
#         3.0139e-02, 3.0269e-02, 3.0291e-02, 3.0381e-02, 3.0410e-02, 3.0631e-02,
#         3.0755e-02, 3.0899e-02, 3.1004e-02, 3.1007e-02, 3.1092e-02, 3.1169e-02,
#         3.1258e-02, 3.1341e-02, 3.1832e-02, 3.2037e-02, 3.2072e-02, 3.2108e-02,
#         3.2125e-02, 3.2399e-02, 3.2495e-02, 3.2541e-02, 3.2557e-02, 3.2622e-02,
#         3.2942e-02, 3.2990e-02, 3.3002e-02, 3.3080e-02, 3.3097e-02, 3.3212e-02,
#         3.3266e-02, 3.3328e-02, 3.3622e-02, 3.3686e-02, 3.3755e-02, 3.4335e-02,
#         3.4497e-02, 3.4565e-02, 3.4703e-02, 3.4928e-02, 3.5160e-02, 3.5165e-02,
#         3.5224e-02, 3.5226e-02, 3.5347e-02, 3.5649e-02, 3.5698e-02, 3.5714e-02,
#         3.5747e-02, 3.5806e-02, 3.5844e-02, 3.5885e-02, 3.6325e-02, 3.6365e-02,
#         3.6414e-02, 3.6429e-02, 3.6434e-02, 3.6451e-02, 3.6567e-02, 3.6584e-02,
#         3.6623e-02, 3.6679e-02, 3.6713e-02, 3.6844e-02, 3.7151e-02, 3.7318e-02,
#         3.7359e-02, 3.7401e-02, 3.7456e-02, 3.7480e-02, 3.7553e-02, 3.7569e-02,
#         3.7610e-02, 3.7677e-02, 3.7737e-02, 3.7769e-02, 3.7867e-02, 3.7887e-02,
#         3.7920e-02, 3.7970e-02, 3.8011e-02, 3.8024e-02, 3.8042e-02, 3.8100e-02,
#         3.8105e-02, 3.8171e-02, 3.8187e-02, 3.8275e-02, 3.8329e-02, 3.8356e-02,
#         3.8566e-02, 3.8571e-02, 3.8598e-02, 3.8707e-02, 3.8713e-02, 3.8728e-02,
#         3.8859e-02, 3.8870e-02, 3.8943e-02, 3.8977e-02, 3.9045e-02, 3.9047e-02,
#         3.9133e-02, 3.9154e-02, 3.9156e-02, 3.9190e-02, 3.9285e-02, 3.9289e-02,
#         3.9304e-02, 3.9317e-02, 3.9368e-02, 3.9430e-02, 3.9438e-02, 3.9689e-02,
#         3.9703e-02, 3.9741e-02, 3.9748e-02, 3.9767e-02, 3.9775e-02, 3.9808e-02,
#         3.9910e-02, 3.9966e-02, 3.9966e-02, 4.0257e-02, 4.0293e-02, 4.0300e-02, # 3.9966e-02 duplicated
#         4.0458e-02, 4.0491e-02, 4.0700e-02, 4.0835e-02, 4.0838e-02, 4.0849e-02,
#         4.0865e-02, 4.0880e-02, 4.1194e-02, 4.1321e-02, 4.1436e-02, 4.1704e-02,
#         4.1989e-02, 4.2097e-02, 4.2406e-02, 4.2480e-02, 4.2507e-02, 4.2538e-02,
#         4.2570e-02, 4.2593e-02, 4.2596e-02, 4.2875e-02, 4.3035e-02, 4.3069e-02,
#         4.3487e-02, 4.3504e-02, 4.3545e-02, 4.3683e-02, 4.3690e-02, 4.4202e-02,
#         4.4214e-02, 4.4239e-02, 4.4366e-02, 4.4431e-02, 4.4776e-02, 4.4873e-02,
#         4.5153e-02, 4.5408e-02, 4.5672e-02, 4.5839e-02, 4.6683e-02, 4.6810e-02,
#         4.7048e-02, 4.7112e-02, 4.7135e-02, 4.7358e-02, 4.7401e-02, 4.7572e-02,
#         4.8153e-02, 4.8227e-02, 4.8293e-02, 4.8736e-02, 4.8914e-02, 4.9197e-02,
#         4.9399e-02, 4.9412e-02, 4.9490e-02, 4.9852e-02, 4.9912e-02, 5.0227e-02,
#         5.0287e-02, 5.0540e-02, 5.0804e-02, 5.1004e-02, 5.1048e-02, 5.1249e-02,
#         5.1263e-02, 5.1470e-02, 5.1611e-02, 5.1777e-02, 5.2054e-02, 5.2082e-02,
#         5.2231e-02, 5.2903e-02, 5.3087e-02, 5.3156e-02, 5.3192e-02, 5.3265e-02,
#         5.3347e-02, 5.3541e-02, 5.3541e-02, 5.3581e-02, 5.3732e-02, 5.3958e-02, # 5.3541e-02 duplicated
#         5.3989e-02, 5.4155e-02, 5.4259e-02, 5.4287e-02, 5.4440e-02, 5.5036e-02,
#         5.5181e-02, 5.6295e-02, 5.6415e-02, 5.6416e-02, 5.6493e-02, 5.6740e-02,
#         5.7032e-02, 5.7142e-02, 5.7285e-02, 5.7296e-02, 5.7348e-02, 5.7378e-02,
#         5.7429e-02, 5.7752e-02, 5.7817e-02, 5.8044e-02, 5.8070e-02, 5.8431e-02,
#         5.8714e-02, 5.9132e-02, 5.9791e-02, 5.9814e-02, 6.0600e-02, 6.0621e-02,
#         6.0789e-02, 6.0879e-02, 6.1254e-02, 6.1492e-02, 6.1878e-02, 6.1989e-02,
#         6.2111e-02, 6.2671e-02, 6.2822e-02, 6.2911e-02, 6.2961e-02, 6.2973e-02,
#         6.3002e-02, 6.3314e-02, 6.3359e-02, 6.3579e-02, 6.3598e-02, 6.3674e-02,
#         6.3979e-02, 6.4312e-02, 6.4743e-02, 6.4877e-02, 6.4995e-02, 6.5003e-02,
#         6.5023e-02, 6.5089e-02, 6.5316e-02, 6.5366e-02, 6.5714e-02, 6.6235e-02,
#         6.6319e-02, 6.6715e-02, 6.6833e-02, 6.7850e-02, 6.8253e-02, 6.8422e-02,
#         6.8528e-02, 6.8723e-02, 6.9201e-02, 6.9235e-02, 6.9262e-02, 6.9379e-02,
#         7.0069e-02, 7.0309e-02, 7.0900e-02, 7.1274e-02, 7.2877e-02, 7.3105e-02,
#         7.3728e-02, 7.3881e-02, 7.3890e-02, 7.4381e-02, 7.4717e-02, 7.5484e-02,
#         7.5795e-02, 7.5829e-02, 7.6401e-02, 7.6429e-02, 7.6953e-02, 7.8132e-02,
#         7.8803e-02, 7.9279e-02, 7.9534e-02, 7.9807e-02, 8.0666e-02, 8.0859e-02,
#         8.1260e-02, 8.3090e-02, 8.3376e-02, 8.4035e-02, 8.4402e-02, 8.4945e-02,
#         8.6176e-02, 8.6178e-02, 8.6620e-02, 8.6738e-02, 8.6776e-02, 8.7009e-02,
#         8.7457e-02, 9.2442e-02, 9.2495e-02, 9.2622e-02, 9.4555e-02, 9.5056e-02,
#         9.6392e-02, 9.7284e-02, 9.7509e-02, 9.7959e-02, 1.0012e-01, 1.0066e-01,
#         1.0101e-01, 1.0163e-01, 1.0240e-01, 1.0251e-01, 1.0340e-01, 1.0536e-01,
#         1.0691e-01, 1.0817e-01, 1.0903e-01, 1.0950e-01, 1.0961e-01, 1.1224e-01,
#         1.1309e-01, 1.1412e-01, 1.1463e-01, 1.1871e-01, 1.1873e-01, 1.1959e-01,
#         1.1984e-01, 1.2330e-01, 1.2654e-01, 1.2710e-01, 1.2893e-01, 1.3132e-01,
#         1.3176e-01, 1.3303e-01, 1.3839e-01, 1.4083e-01, 1.5848e-01, 1.6255e-01,
#         1.6625e-01, 1.7775e-01, 2.2206e-01, 2.2377e-01] # Neuron=26


# absolute_normalized_hidden_output = [1.8856e-05, 3.7243e-05, 5.2667e-05, 2.3186e-04, 3.1989e-04, 5.3054e-04,
#         6.8351e-04, 7.5729e-04, 8.2523e-04, 9.3408e-04, 1.2013e-03, 1.2321e-03,
#         1.2657e-03, 1.3091e-03, 1.3403e-03, 1.4740e-03, 1.5030e-03, 1.5620e-03,
#         1.7247e-03, 1.7408e-03, 1.9684e-03, 2.0495e-03, 2.1847e-03, 2.2393e-03,
#         2.3916e-03, 2.4431e-03, 2.5593e-03, 2.6467e-03, 2.7207e-03, 2.7573e-03,
#         2.7781e-03, 2.9462e-03, 2.9926e-03, 3.0060e-03, 3.0782e-03, 3.1310e-03,
#         3.1494e-03, 3.3772e-03, 3.4011e-03, 3.4363e-03, 3.5314e-03, 3.5326e-03,
#         3.6593e-03, 3.6609e-03, 3.8366e-03, 3.9114e-03, 4.0238e-03, 4.0911e-03,
#         4.0916e-03, 4.5519e-03, 4.7169e-03, 4.8057e-03, 4.9765e-03, 5.1886e-03,
#         5.3313e-03, 5.3562e-03, 5.9760e-03, 6.0811e-03, 6.2255e-03, 6.2865e-03,
#         6.3775e-03, 6.5619e-03, 6.6985e-03, 6.6999e-03, 7.0718e-03, 7.0803e-03,
#         7.1842e-03, 7.2353e-03, 7.3043e-03, 7.3091e-03, 7.3388e-03, 7.3393e-03,
#         7.5571e-03, 7.6184e-03, 7.6287e-03, 7.6447e-03, 7.8899e-03, 8.0180e-03,
#         8.0240e-03, 8.0801e-03, 8.1316e-03, 8.5297e-03, 8.5675e-03, 8.6384e-03,
#         8.7734e-03, 8.9977e-03, 9.1490e-03, 9.4020e-03, 9.5911e-03, 9.6423e-03,
#         9.7022e-03, 9.8746e-03, 9.9184e-03, 1.0417e-02, 1.0515e-02, 1.0719e-02,
#         1.0778e-02, 1.0779e-02, 1.0808e-02, 1.0867e-02, 1.0896e-02, 1.1161e-02,
#         1.1542e-02, 1.1774e-02, 1.2010e-02, 1.2165e-02, 1.2306e-02, 1.2322e-02,
#         1.2387e-02, 1.2564e-02, 1.2905e-02, 1.3174e-02, 1.3215e-02, 1.3228e-02,
#         1.3338e-02, 1.3371e-02, 1.3403e-02, 1.3641e-02, 1.3734e-02, 1.3766e-02,
#         1.3821e-02, 1.3863e-02, 1.4008e-02, 1.4531e-02, 1.4694e-02, 1.4851e-02,
#         1.5041e-02, 1.5097e-02, 1.5174e-02, 1.5370e-02, 1.5373e-02, 1.5492e-02,
#         1.5567e-02, 1.5953e-02, 1.6105e-02, 1.6223e-02, 1.6477e-02, 1.6561e-02,
#         1.6592e-02, 1.6692e-02, 1.6801e-02, 1.6966e-02, 1.7209e-02, 1.7246e-02,
#         1.7428e-02, 1.7480e-02, 1.7553e-02, 1.7786e-02, 1.7936e-02, 1.8089e-02,
#         1.8483e-02, 1.8657e-02, 1.8751e-02, 1.8859e-02, 1.8869e-02, 1.9071e-02,
#         1.9076e-02, 1.9112e-02, 1.9172e-02, 1.9502e-02, 1.9541e-02, 1.9675e-02,
#         1.9787e-02, 1.9849e-02, 1.9897e-02, 2.0150e-02, 2.0165e-02, 2.0346e-02,
#         2.0347e-02, 2.0354e-02, 2.0456e-02, 2.0529e-02, 2.0624e-02, 2.0673e-02,
#         2.1099e-02, 2.1207e-02, 2.1319e-02, 2.1659e-02, 2.1748e-02, 2.2180e-02,
#         2.2237e-02, 2.2301e-02, 2.2327e-02, 2.2364e-02, 2.2370e-02, 2.2655e-02,
#         2.2751e-02, 2.2778e-02, 2.2786e-02, 2.2849e-02, 2.2860e-02, 2.2897e-02,
#         2.3023e-02, 2.3214e-02, 2.3229e-02, 2.3274e-02, 2.3298e-02, 2.3772e-02,
#         2.3921e-02, 2.4071e-02, 2.4340e-02, 2.4496e-02, 2.4554e-02, 2.4661e-02,
#         2.4766e-02, 2.4809e-02, 2.4835e-02, 2.4904e-02, 2.4957e-02, 2.5043e-02,
#         2.5069e-02, 2.5338e-02, 2.5442e-02, 2.5480e-02, 2.5503e-02, 2.5518e-02,
#         2.5607e-02, 2.5668e-02, 2.5683e-02, 2.5867e-02, 2.5879e-02, 2.6044e-02,
#         2.6099e-02, 2.6189e-02, 2.6199e-02, 2.6242e-02, 2.6380e-02, 2.6400e-02,
#         2.6469e-02, 2.6580e-02, 2.6616e-02, 2.6647e-02, 2.6740e-02, 2.6772e-02,
#         2.6842e-02, 2.7033e-02, 2.7258e-02, 2.7273e-02, 2.7331e-02, 2.7418e-02,
#         2.7545e-02, 2.7548e-02, 2.7747e-02, 2.7757e-02, 2.7780e-02, 2.7880e-02,
#         2.7979e-02, 2.8065e-02, 2.8184e-02, 2.8303e-02, 2.8315e-02, 2.8610e-02,
#         2.8671e-02, 2.8781e-02, 2.8804e-02, 2.8949e-02, 2.9025e-02, 2.9033e-02,
#         2.9219e-02, 2.9227e-02, 2.9322e-02, 2.9335e-02, 2.9406e-02, 2.9546e-02,
#         2.9696e-02, 2.9765e-02, 3.0092e-02, 3.0847e-02, 3.0978e-02, 3.1035e-02,
#         3.1295e-02, 3.1582e-02, 3.2074e-02, 3.2125e-02, 3.2183e-02, 3.2300e-02,
#         3.2311e-02, 3.2405e-02, 3.2437e-02, 3.2572e-02, 3.2844e-02, 3.3314e-02,
#         3.3473e-02, 3.3507e-02, 3.3547e-02, 3.3552e-02, 3.3632e-02, 3.3635e-02,
#         3.3683e-02, 3.3877e-02, 3.3891e-02, 3.3923e-02, 3.3942e-02, 3.3974e-02,
#         3.3997e-02, 3.4021e-02, 3.4035e-02, 3.4153e-02, 3.4241e-02, 3.4301e-02,
#         3.4537e-02, 3.4757e-02, 3.4776e-02, 3.4907e-02, 3.4923e-02, 3.4971e-02,
#         3.5296e-02, 3.5378e-02, 3.5390e-02, 3.5483e-02, 3.5580e-02, 3.5748e-02,
#         3.5749e-02, 3.5752e-02, 3.5995e-02, 3.6019e-02, 3.6461e-02, 3.6873e-02,
#         3.6887e-02, 3.6999e-02, 3.7040e-02, 3.7057e-02, 3.7342e-02, 3.7450e-02,
#         3.7488e-02, 3.7502e-02, 3.7586e-02, 3.7664e-02, 3.7782e-02, 3.7816e-02,
#         3.7889e-02, 3.7982e-02, 3.8293e-02, 3.8303e-02, 3.8479e-02, 3.8546e-02,
#         3.8610e-02, 3.8794e-02, 3.8917e-02, 3.8965e-02, 3.9395e-02, 3.9446e-02,
#         3.9512e-02, 3.9537e-02, 3.9600e-02, 3.9651e-02, 3.9682e-02, 3.9711e-02,
#         3.9756e-02, 3.9957e-02, 4.0127e-02, 4.0348e-02, 4.0370e-02, 4.0393e-02,
#         4.0454e-02, 4.0495e-02, 4.0496e-02, 4.0668e-02, 4.0915e-02, 4.1006e-02,
#         4.1172e-02, 4.1346e-02, 4.1739e-02, 4.1808e-02, 4.1931e-02, 4.1949e-02,
#         4.2028e-02, 4.2125e-02, 4.2129e-02, 4.2352e-02, 4.2412e-02, 4.2509e-02,
#         4.2551e-02, 4.2610e-02, 4.2671e-02, 4.2823e-02, 4.2903e-02, 4.2951e-02,
#         4.3018e-02, 4.3132e-02, 4.3230e-02, 4.3302e-02, 4.3465e-02, 4.3493e-02,
#         4.3538e-02, 4.3671e-02, 4.3712e-02, 4.3739e-02, 4.3779e-02, 4.3901e-02,
#         4.4204e-02, 4.4232e-02, 4.4302e-02, 4.4344e-02, 4.4364e-02, 4.4548e-02,
#         4.4987e-02, 4.5048e-02, 4.5299e-02, 4.5456e-02, 4.5464e-02, 4.5531e-02,
#         4.5713e-02, 4.5772e-02, 4.6053e-02, 4.6131e-02, 4.6274e-02, 4.6558e-02,
#         4.6627e-02, 4.6760e-02, 4.6934e-02, 4.6959e-02, 4.6970e-02, 4.7017e-02,
#         4.7159e-02, 4.7390e-02, 4.7500e-02, 4.7531e-02, 4.7537e-02, 4.7628e-02,
#         4.7727e-02, 4.8100e-02, 4.8121e-02, 4.8171e-02, 4.8235e-02, 4.8247e-02,
#         4.8249e-02, 4.8251e-02, 4.8254e-02, 4.8338e-02, 4.8338e-02, 4.8415e-02, # 4.8338e-02 duplicated
#         4.8424e-02, 4.8437e-02, 4.8558e-02, 4.9522e-02, 5.0190e-02, 5.0202e-02,
#         5.0321e-02, 5.0481e-02, 5.0695e-02, 5.0728e-02, 5.0802e-02, 5.0823e-02,
#         5.0862e-02, 5.0952e-02, 5.1207e-02, 5.1307e-02, 5.1535e-02, 5.1586e-02,
#         5.1755e-02, 5.2175e-02, 5.2309e-02, 5.2417e-02, 5.2952e-02, 5.3513e-02,
#         5.3961e-02, 5.4192e-02, 5.4264e-02, 5.4363e-02, 5.4901e-02, 5.5251e-02,
#         5.5372e-02, 5.5642e-02, 5.5675e-02, 5.5848e-02, 5.5884e-02, 5.6117e-02,
#         5.6128e-02, 5.6173e-02, 5.6465e-02, 5.6639e-02, 5.6794e-02, 5.6834e-02,
#         5.6937e-02, 5.6968e-02, 5.7085e-02, 5.7992e-02, 5.8174e-02, 5.8214e-02,
#         5.8506e-02, 5.8843e-02, 5.9154e-02, 5.9172e-02, 5.9174e-02, 5.9193e-02,
#         5.9675e-02, 5.9864e-02, 6.1037e-02, 6.1067e-02, 6.1446e-02, 6.1470e-02,
#         6.1706e-02, 6.1777e-02, 6.2368e-02, 6.2372e-02, 6.2417e-02, 6.2438e-02,
#         6.2531e-02, 6.2789e-02, 6.3076e-02, 6.3162e-02, 6.4002e-02, 6.4103e-02,
#         6.4186e-02, 6.4411e-02, 6.4548e-02, 6.5095e-02, 6.5672e-02, 6.5910e-02,
#         6.6240e-02, 6.6591e-02, 6.6609e-02, 6.6611e-02, 6.7318e-02, 6.7705e-02,
#         6.7961e-02, 6.8188e-02, 6.9092e-02, 6.9166e-02, 6.9699e-02, 6.9794e-02,
#         7.0084e-02, 7.0884e-02, 7.0986e-02, 7.1262e-02, 7.1341e-02, 7.1712e-02,
#         7.2537e-02, 7.2955e-02, 7.3018e-02, 7.3329e-02, 7.4520e-02, 7.4781e-02,
#         7.4952e-02, 7.5023e-02, 7.5037e-02, 7.6064e-02, 7.6147e-02, 7.6788e-02,
#         7.6868e-02, 7.6927e-02, 7.8095e-02, 7.8327e-02, 7.8423e-02, 7.8850e-02,
#         7.8878e-02, 7.9249e-02, 7.9366e-02, 8.1180e-02, 8.1271e-02, 8.1635e-02,
#         8.2037e-02, 8.2550e-02, 8.2936e-02, 8.3952e-02, 8.4385e-02, 8.4385e-02, # 8.4385e-02 duplicated
#         8.4405e-02, 8.4765e-02, 8.5884e-02, 8.6128e-02, 8.6413e-02, 8.6574e-02,
#         8.6622e-02, 8.6689e-02, 8.6777e-02, 8.6979e-02, 8.7041e-02, 8.7931e-02,
#         8.8473e-02, 8.9613e-02, 9.0371e-02, 9.2112e-02, 9.2224e-02, 9.2520e-02,
#         9.2945e-02, 9.3234e-02, 9.3235e-02, 9.4573e-02, 9.5067e-02, 9.5205e-02,
#         9.5602e-02, 9.6012e-02, 9.6312e-02, 9.7011e-02, 9.7011e-02, 9.7589e-02, # 9.7011e-02 duplicated
#         9.8761e-02, 9.9212e-02, 9.9750e-02, 1.0031e-01, 1.0287e-01, 1.0369e-01,
#         1.0394e-01, 1.0430e-01, 1.0527e-01, 1.0610e-01, 1.0610e-01, 1.0694e-01, # 1.0610e-01 duplicated
#         1.0742e-01, 1.0955e-01, 1.1241e-01, 1.1608e-01, 1.1891e-01, 1.2032e-01,
#         1.2103e-01, 1.2276e-01, 1.2735e-01, 1.2914e-01, 1.3004e-01, 1.3035e-01,
#         1.3053e-01, 1.3563e-01, 1.3800e-01, 1.4097e-01, 1.4187e-01, 1.4195e-01,
#         1.4879e-01, 1.5189e-01, 1.5581e-01, 1.6992e-01, 1.7781e-01, 2.2762e-01]
#
# neuron_number_modify = torch.arange(0, len(absolute_normalized_hidden_output), 1)
# save_pth = f"Result/Model/{encoder_type}{input_size}_{bits}/{Model_type}_{neuron_number}_deleted_{device}/"
# for i in range(len((absolute_normalized_hidden_output))):
#     model_modified = modify(bits, input_size, absolute_normalized_hidden_output[i], neuron_number, origin_model, parameter, fc_model_pth, save_pth)
# print("Model Modify Finished")

# # model inspect:
# neuron_number_inspect = 7
# threshold_inspect = 0.1 # normalized
#
# modified_model_pth = f"{encoder_type}/Result/Model/{Model_type}_modified_threshold{threshold_inspect}_{device}/{Model_type}_model_modified_hiddenlayer{neuron_number_inspect}_BER0.pth"
# # print("modified model:", torch.load(modified_model_pth))

Model_type = "SLNN"
bits = 10
input_size = 26
output_size = torch.pow(torch.tensor(2), bits)
encoder_type = "Parity"
device = "cpu"
neuron_number = 26
origin_model = SingleLabelNNDecoder1
parameter = "hidden.weight"
edge_delete = 619

model_pth = f"Result/Model/{encoder_type}{input_size}_{bits}/{neuron_number}_ft_{device}/{Model_type}_deleted{edge_delete}_trained.pth"

# position = one_positions = (mask == 1).nonzero()
position = torch.tensor([
    # [ 0,  4], # 0
    # [ 0,  6], # 1
    # [ 1,  5], # 2
    # [ 1, 21], # 3
    [ 2,  3], # 4
    # [ 3,  4], # 5
    # [ 3,  7], # 6
    [ 3, 13], # 7
    # [ 3, 21], # 8
    # [ 4, 14], # 9
    # [ 4, 23], # 10
    [ 5,  5], # 11
    # [ 6, 16], # 12
    [ 6, 17], # 13
    # [ 6, 19], # 14
    # [ 7, 15], # 15
    # [ 7, 18], # 16
    # [ 8, 13], # 17
    # [ 9,  0], # 18
    # [ 9, 18], # 19
    # [10,  6], # 20
    # [11, 15], # 21
    # [11, 21], # 22
    # [12,  0], # 23
    # [12,  2], # 24
    # [13,  2], # 25
    # [13,  3], # 26
    [13, 24], # 27
    # [14,  6], # 28
    [15,  8], # 29
    # [15, 16], # 30
    # [16, 19], # 31
    # [16, 25], # 32
    # [17,  1], # 33
    # [17, 17], # 34
    # [17, 20], # 35
    # [18,  5], # 36
    # [18, 20], # 37
    # [19,  2], # 38
    # [19,  9], # 39
    # [19, 13], # 40
    # [20, 22], # 41
    # [20, 24], # 42
    # [21,  2], # 43
    # [21, 11], # 44
    # [21, 13], # 45
    # [22, 19], # 46
    # [22, 25], # 47
    # [23,  3], # 48
    # [23, 21], # 49
    # [24,  5], # 50
    # [24,  9], # 51
    # [24, 20], # 52
    # [24, 23], # 53
    # [25,  4], # 54
    # [25,  7], # 55
    # [25, 16] # 56
                         ]) # To filter out the edge that has been deleted

save_pth = f"Result/Model/{encoder_type}{input_size}_{bits}/{neuron_number}_ft_{device}/"

# for i in range(len((position))):
#     model_modified = modify_exact(bits, input_size, position[i], neuron_number, origin_model, parameter, model_pth, save_pth, i)

i = 0
model_modified = modify_exact_all(bits, input_size, position, neuron_number, origin_model, parameter, model_pth,
                                  save_pth, i)

print("Model Modify Finished")
