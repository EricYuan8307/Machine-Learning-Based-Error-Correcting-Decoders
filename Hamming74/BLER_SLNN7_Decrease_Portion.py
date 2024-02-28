import torch
import numpy as np
import os
from datetime import datetime

from Hamming74.reduce_mask import MaskMatrix
from Encode.Generator import generator
from Encode.Modulator import bpsk_modulator
from Decode.NNDecoder import SingleLabelNNDecoder_nonfully
from Transmit.noise import AWGN
from Metric.ErrorRate import calculate_bler
from Transmit.NoiseMeasure import NoiseMeasure
from Decode.Converter import DecimaltoBinary

from generating import all_codebook, SLNN_D2B_matrix
from Encode.Encoder import PCC_encoders

def SLNNDecoder(nr_codeword, bits, encoded, snr_dB, model, model_pth, device):
    encoder_matrix, decoder_matrix, SoftDecisionMLMatrix = all_codebook(bits, encoded, device)
    SLNN_Matrix = SLNN_D2B_matrix(bits, device)

    encoder = PCC_encoders(encoder_matrix)
    convertor = DecimaltoBinary(SLNN_Matrix)

    bits_info = generator(nr_codeword, bits, device)  # Code Generator
    encoded_codeword = encoder(bits_info)  # Hamming(7,4) Encoder
    modulated_signal = bpsk_modulator(encoded_codeword)  # Modulate signal
    noised_signal = AWGN(modulated_signal, snr_dB, device)  # Add Noise

    practical_snr = NoiseMeasure(noised_signal, modulated_signal, bits, encoded)

    # use SLNN model:
    model.eval()
    model.load_state_dict(torch.load(model_pth))

    SLNN_result = model(noised_signal)
    SLNN_decimal = torch.argmax(SLNN_result, dim=2)

    SLNN_binary = convertor(SLNN_decimal)


    return SLNN_binary, bits_info, practical_snr

def estimation(num, bits, encoded, SNR_opt_NN, SLNN_hidden_size, model_pth, mask, edge_delete, device):
    # Single-label Neural Network:
    output_size = torch.pow(torch.tensor(2), bits)

    model = SingleLabelNNDecoder_nonfully(encoded, SLNN_hidden_size, output_size, mask).to(device)
    SLNN_final, bits_info, snr_measure = SLNNDecoder(num, bits, encoded, SNR_opt_NN, model, model_pth, device)

    BLER_SLNN, error_num_SLNN = calculate_bler(SLNN_final, bits_info) # BER calculation

    if error_num_SLNN < 100:
        num += 1000000
        print(f"the code number is {num}")

    else:
        print(f"SLNN edge deleted{edge_delete}: When SNR is {snr_measure} and signal number is {num}, error number is {error_num_SLNN} and BLER is {BLER_SLNN}")

    return BLER_SLNN


def main():
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.backends.cuda.is_available()
    #                 else torch.device("cpu")))
    device = torch.device("cpu")
    # device = torch.device("cuda")

    # Hyperparameters for SLNN neuron=7
    num = int(2e7)
    bits = 4
    encoded = 7
    SLNN_hidden_size = 7
    edge_delete = [9, 14, 19, 24, 29, 34, 39, 43]

    masks = MaskMatrix(device)


    SNR_opt_NN = torch.tensor(8, dtype=torch.int, device=device)
    SNR_opt_NN = SNR_opt_NN + 10 * torch.log10(torch.tensor(bits / encoded, dtype=torch.float)) # for SLNN article

    parameter = "hidden.weight"

    for i in range(len(edge_delete)):
        mask = masks(edge_delete[i], encoded, SLNN_hidden_size)
        load_pth = f"Result/Model/SLNN_decrease_{parameter}_{device}/SLNN_model_hiddenlayer{edge_delete[i]}_BER0.pth"
        result_all = estimation(num, bits, encoded, SNR_opt_NN, SLNN_hidden_size, load_pth, mask, edge_delete[i], device)
    # directory_path = "Result/BLER"
    #
    # # Create the directory if it doesn't exist
    # if not os.path.exists(directory_path):
    #     os.makedirs(directory_path)
    #
    # current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # csv_filename = f"BLER_result_{current_time}.csv"
    # full_csv_path = os.path.join(directory_path, csv_filename)
    # np.savetxt(full_csv_path, result_all, delimiter=', ')


if __name__ == "__main__":
    main()