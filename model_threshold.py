import torch
import os
from Decode.NNDecoder import SingleLabelNNDecoder

def modify(origin_size, input_size, threshold, Model_type, neuron_number, encoder_type, origin_model, parameter, device):
    output_size = torch.pow(torch.tensor(2), origin_size)

    origin_model_pth = f"{encoder_type}/Result/Model/{Model_type}_{device}/{Model_type}_model_hiddenlayer{neuron_number}_BER0.pth"
    save_pth = f"{encoder_type}/Result/Model/{Model_type}_modified_threshold{threshold}_{device}/"

    # Assuming you have the model class defined somewhere
    model = origin_model(input_size, neuron_number, output_size)
    model.load_state_dict(torch.load(origin_model_pth))

    for name, param in model.named_parameters():
        if name == parameter:
            # Apply thresholding to the absoluted normalized weight values
            with (torch.no_grad()):  # Ensure that these operations don't track gradients
                normalized = torch.div(torch.abs(param.data), torch.sum(torch.abs(param.data), dim=1).unsqueeze(1))
                param.data = torch.where(normalized < threshold, torch.zeros_like(param.data), param.data)

    # Create the directory if it doesn't exist
    os.makedirs(save_pth, exist_ok=True)
    torch.save(model.state_dict(),
               f"{save_pth}{Model_type}_model_modified_hiddenlayer{neuron_number}_BER0.pth")

    model_modified = torch.load(f'{save_pth}{Model_type}_model_modified_hiddenlayer{neuron_number}_BER0.pth')
    return model_modified


def loadpara(origin_size, input_size, Model_type, neuron_number, encoder_type, origin_model, device):
    output_size = torch.pow(torch.tensor(2), origin_size)

    origin_model_pth = f"{encoder_type}/Result/Model/{Model_type}_{device}/{Model_type}_model_hiddenlayer{neuron_number}_BER0.pth"
    model = torch.load(origin_model_pth)
    print("model parameters:",model)

    # Assuming you have the model class defined somewhere
    model = origin_model(input_size, neuron_number, output_size)
    model.load_state_dict(torch.load(origin_model_pth))

    for name, param in model.named_parameters():
        if name == "hidden.weight":
            normalized = torch.div(torch.abs(param.data), torch.sum(torch.abs(param.data), dim=1).unsqueeze(1))
            print(f"normalized {name}:", normalized)


# Model Check
Model_type = "SLNN"
origin_size = 4
input_size = 7
output_size = torch.pow(torch.tensor(2), origin_size)
encoder_type = "Hamming74"
device = "cpu"
neuron_number = torch.arange(0, 101, 1)

# # Model modify:
threshold = 0.05 # normalized
parameter = "hidden.weight"

origin_model = SingleLabelNNDecoder
# model_para = loadpara(origin_size, input_size, Model_type, neuron_number, encoder_type, origin_model, device)

# # modified result
for i in range(len(neuron_number)):
    model_modified = modify(origin_size, input_size, threshold, Model_type, neuron_number[i], encoder_type, origin_model, parameter, device)
# modified_model_pth = f"{encoder_type}/Result/Model/{Model_type}_modified_{device}/{Model_type}_model_modified_hiddenlayer{neuron_number}_BER0.pth"
# print("modified model:", torch.load(modified_model_pth))