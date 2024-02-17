import torch
import os
from Decode.NNDecoder import SingleLabelNNDecoder, MultiLabelNNDecoder1, MultiLabelNNDecoder2

def modify(origin_size, input_size, threshold, Model_type, neuron_number, encoder_type, origin_model):
    output_size = torch.pow(torch.tensor(2), origin_size)

    origin_model_pth = f"{encoder_type}/Result/Model/{Model_type}_CPU/{Model_type}_model_hiddenlayer{neuron_number}_BER0.pth"
    save_pth = f"{encoder_type}/Result/Model/{Model_type}_modified_CPU/"

    # Assuming you have the model class defined somewhere
    model = origin_model(input_size, neuron_number, output_size)
    model.load_state_dict(torch.load(origin_model_pth))

    for name, param in model.named_parameters():
        # Apply thresholding to the weights
        with torch.no_grad():  # Ensure that these operations don't track gradients
            param.data = torch.where(abs(param.data) < threshold, torch.zeros_like(param.data), param.data)

    # Create the directory if it doesn't exist
    os.makedirs(save_pth, exist_ok=True)
    torch.save(model.state_dict(),
               f"{save_pth}{Model_type}_model_modified_hiddenlayer{neuron_number}_BER0_threshold{threshold}.pth")


def loadpara(Model_type, neuron_number, encoder_type):
    model = torch.load(f'{encoder_type}/Result/Model/{Model_type}_CPU/{Model_type}_model_hiddenlayer{neuron_number}_BER0.pth')

    return model


# Model Check
Model_type = "SLNN"
neuron_number = torch.arange(0, 101, 1)
encoder_type = "Hamming74"

# Model modify:
origin_size = 4
input_size = 7
output_size = torch.pow(torch.tensor(2), origin_size)
threshold = 0.01
origin_model = SingleLabelNNDecoder

for i in range(len(neuron_number)):
    model_para = loadpara(Model_type, neuron_number[i], encoder_type)
    print(f"{neuron_number}neruons model Parameters:",model_para)

    modify(origin_size, input_size, threshold, Model_type, neuron_number[i], encoder_type, origin_model)