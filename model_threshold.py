import torch
import os
from Decode.NNDecoder import SingleLabelNNDecoder

def normalize(data):
    normalized = torch.div(torch.abs(data), torch.sum(torch.abs(data), dim=1).unsqueeze(1))
    return normalized

def modify(origin_size, input_size, threshold, Model_type, neuron_number, encoder_type, origin_model, parameter, device):
    output_size = torch.pow(torch.tensor(2), origin_size)

    origin_model_pth = f"{encoder_type}/Result/Model/{Model_type}_{device}/{Model_type}_model_hiddenlayer{neuron_number}_BER0.pth"
    save_pth = f"{encoder_type}/Result/Model/{Model_type}_modified_threshold{threshold}_{device}/" # for all SLNN model
    save_pth = f"{encoder_type}/Result/Model/{Model_type}_modified_neuron{neuron_number}_{device}_{parameter}/" # exclusive for Neuron=7

    # Assuming you have the model class defined somewhere
    model = origin_model(input_size, neuron_number, output_size)
    model.load_state_dict(torch.load(origin_model_pth))

    for name, param in model.named_parameters():
        if name == parameter:
            # Apply thresholding to the absoluted normalized weight values
            with (torch.no_grad()):  # Ensure that these operations don't track gradients
                normalized = normalize(param.data)
                param.data = torch.where(normalized < threshold, torch.zeros_like(param.data), param.data)

    # Create the directory if it doesn't exist
    os.makedirs(save_pth, exist_ok=True)
    # torch.save(model.state_dict(), f"{save_pth}{Model_type}_model_modified_hiddenlayer{neuron_number}_BER0.pth") # for normal model
    torch.save(model.state_dict(), f"{save_pth}{Model_type}_model_modified_hiddenlayer{neuron_number}_threshold{threshold}_BER0.pth") # exclusive for neuron=7

    # model_modified = torch.load(f'{save_pth}{Model_type}_model_modified_hiddenlayer{neuron_number}_BER0.pth')
    # return model_modified


def loadpara(origin_size, input_size, Model_type, neuron_number, encoder_type, origin_model, device):
    output_size = torch.pow(torch.tensor(2), origin_size)

    origin_model_pth = f"{encoder_type}/Result/Model/{Model_type}_{device}/{Model_type}_model_hiddenlayer{neuron_number}_BER0.pth"
    model = torch.load(origin_model_pth)
    # print("model parameters:",model)

    # Assuming you have the model class defined somewhere
    model = origin_model(input_size, neuron_number, output_size)
    model.load_state_dict(torch.load(origin_model_pth))

    for name, param in model.named_parameters():
        if name in ("hidden.weight", "output.weight"):
            normalized = torch.div(torch.abs(param.data), torch.sum(torch.abs(param.data), dim=1).unsqueeze(1))
            print(f"normalized {name}:", normalized)


# Model Check
Model_type = "SLNN"
origin_size = 4
input_size = 7
output_size = torch.pow(torch.tensor(2), origin_size)
encoder_type = "Hamming74"
device = "cpu"
neuron_number = 7
origin_model = SingleLabelNNDecoder

# Check original model:
# model_para = loadpara(origin_size, input_size, Model_type, neuron_number, encoder_type, origin_model, device)

# # Model modify:
# threshold = 0.05 # normalized
threshold = torch.arange(0, 0.4, 0.1)
parameter = "hidden.weight"
# parameter = "hidden.weight", "output.weight"
# neuron_number_modify = torch.arange(0, 101, 1)
neuron_number_modify = 7

for i in range(len(threshold)):
    model_modified = modify(origin_size, input_size, threshold[i], Model_type, neuron_number_modify, encoder_type, origin_model, parameter, device)
print("Model Modify Finished")

# model inspect:
neuron_number_inspect = 7
threshold_inspect = 0.1 # normalized

# modified_model_pth = f"{encoder_type}/Result/Model/{Model_type}_modified_threshold{threshold_inspect}_{device}/{Model_type}_model_modified_hiddenlayer{neuron_number_inspect}_BER0.pth"
# print("modified model:", torch.load(modified_model_pth))