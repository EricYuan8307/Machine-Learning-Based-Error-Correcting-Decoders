import torch
import os
from Decode.NNDecoder import MultiLabelNNDecoder_N

def normalize_abs(data):
    normalized = torch.div(torch.abs(data), torch.sum(torch.abs(data), dim=1).unsqueeze(1))
    return normalized


def modify(origin_size, input_size, threshold, neuron_number, origin_model, parameter, origin_model_pth, save_pth):
    # Assuming you have the model class defined somewhere
    model = origin_model(input_size, neuron_number, origin_size)
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

    # Assuming you have the model class defined somewhere
    model = origin_model(input_size, neuron_number, origin_size)
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


# Model_type = "MLNN"
# bits = 4
# input_size = 7
# output_size = torch.pow(torch.tensor(2), bits)
# encoder_type = "Hamming"
# device = "cpu"
# neuron_number = 16
# origin_model = SingleLabelNNDecoder1
# parameter = "output.weight"
# # edge_delete = 42
#
# model_pth = f"Result/Model/{encoder_type}{input_size}_{bits}/{Model_type}_{device}/{Model_type}_hiddenlayer{neuron_number}.pth"
#
# # Model modify:
# absolute_normalized_hidden_output = [
#     0.0536, 0.0538, 0.0549, 0.0551, 0.0551, 0.0557, 0.0557, 0.0558, 0.0558, 0.0563, 0.0564, 0.0566, 0.0567, 0.0571,
#     0.0575, 0.0588, 0.0593, 0.0594, 0.0596, 0.0598, 0.0598, 0.0605, 0.0610, 0.0611, 0.0612, 0.0612, 0.0619, 0.0619,
#     0.0619, 0.0621, 0.0624, 0.0625, 0.0625, 0.0626, 0.0630, 0.0631, 0.0631, 0.0633, 0.0635, 0.0636, 0.0637, 0.0644,
#     0.0647, 0.0650, 0.0651, 0.0652, 0.0654, 0.0660, 0.0675, 0.0676, 0.0676, 0.0677, 0.0685, 0.0685, 0.0687, 0.0689,
#     0.0691, 0.0691, 0.0692, 0.0698, 0.0702, 0.0706, 0.0711, 0.0712]
#
# neuron_number_modify = torch.arange(0, len(absolute_normalized_hidden_output), 1)
# save_pth = f"Result/Model/{encoder_type}{input_size}_{bits}/{neuron_number}_ft_{device}/"
# for i in range(len((absolute_normalized_hidden_output))):
#     model_modified = modify(bits, input_size, absolute_normalized_hidden_output[i], neuron_number, origin_model, parameter, model_pth, save_pth)
# print("Model Modify Finished")


Model_type = "MLNN"
bits = 4
input_size = 7
encoder_type = "Hamming"
device = "cpu"
neuron_number = 16
origin_model = MultiLabelNNDecoder_N
parameter = "output.weight"
# edge_delete = 550
model_pth = f"Result/Model/{encoder_type}{input_size}_{bits}/{Model_type}_{device}/{Model_type}_hiddenlayer{neuron_number}.pth"
# model_pth = f"Result/Model/{encoder_type}{input_size}_{bits}/{neuron_number}_ft_{device}/{Model_type}_deleted{edge_delete}_trained.pth"

position = torch.tensor([[ 0,  0],
        [ 0,  1],
        # [ 0,  2],
        # [ 0,  3],
        # [ 0,  4],
        # [ 0,  5],
        # [ 0,  6],
        # [ 1,  0],
        # [ 1,  1],
        # [ 1,  2],
        # [ 1,  3],
        # [ 1,  4],
        # [ 1,  5],
        # [ 1,  6],
        # [ 2,  0],
        # [ 2,  1],
        # [ 2,  2],
        # [ 2,  3],
        # [ 2,  4],
        # [ 2,  5],
        # [ 2,  6],
        # [ 3,  0],
        # [ 3,  1],
        # [ 3,  2],
        # [ 3,  3],
        # [ 3,  4],
        # [ 3,  5],
        # [ 3,  6],
        # [ 4,  0],
        # [ 4,  1],
        # [ 4,  2],
        # [ 4,  3],
        # [ 4,  4],
        # [ 4,  5],
        # [ 4,  6],
        # [ 5,  0],
        # [ 5,  1],
        # [ 5,  2],
        # [ 5,  3],
        # [ 5,  4],
        # [ 5,  5],
        # [ 5,  6],
        # [ 6,  0],
        # [ 6,  1],
        # [ 6,  2],
        # [ 6,  3],
        # [ 6,  4],
        # [ 6,  5],
        # [ 6,  6],
        # [ 7,  0],
        # [ 7,  1],
        # [ 7,  2],
        # [ 7,  3],
        # [ 7,  4],
        # [ 7,  5],
        # [ 7,  6],
        # [ 8,  0],
        # [ 8,  1],
        # [ 8,  2],
        # [ 8,  3],
        # [ 8,  4],
        # [ 8,  5],
        # [ 8,  6],
        # [ 9,  0],
        # [ 9,  1],
        # [ 9,  2],
        # [ 9,  3],
        # [ 9,  4],
        # [ 9,  5],
        # [ 9,  6],
        # [10,  0],
        # [10,  1],
        # [10,  2],
        # [10,  3],
        # [10,  4],
        # [10,  5],
        # [10,  6],
        # [11,  0],
        # [11,  1],
        # [11,  2],
        # [11,  3],
        # [11,  4],
        # [11,  5],
        # [11,  6],
        # [12,  0],
        # [12,  1],
        # [12,  2],
        # [12,  3],
        # [12,  4],
        # [12,  5],
        # [12,  6],
        # [13,  0],
        # [13,  1],
        # [13,  2],
        # [13,  3],
        # [13,  4],
        # [13,  5],
        # [13,  6],
        # [14,  0],
        # [14,  1],
        # [14,  2],
        # [14,  3],
        # [14,  4],
        # [14,  5],
        # [14,  6],
        # [15,  0],
        # [15,  1],
        # [15,  2],
        # [15,  3],
        # [15,  4],
        # [15,  5],
        # [15,  6]
                         ])

save_pth = f"Result/Model/{encoder_type}{input_size}_{bits}/{neuron_number}_ft_{device}/"

# for i in range(len((position))):
#     model_modified = modify_exact(bits, input_size, position[i], neuron_number, origin_model, parameter, model_pth, save_pth, i)

i = 0
model_modified = modify_exact_all(bits, input_size, position, neuron_number, origin_model, parameter, model_pth, save_pth, i)

print("Model Modify Finished")
