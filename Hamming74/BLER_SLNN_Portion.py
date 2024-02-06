import torch
import numpy as np
import os
from datetime import datetime

from Encode.Generator import generator
from Encode.Modulator import bpsk_modulator
from Encode.Encoder import hamming74_encoder
from Decoder.NNDecoder import SingleLabelNNDecoder
from Transmit.noise import AWGN
from Metric.ErrorRate import calculate_bler
from Transmit.NoiseMeasure import NoiseMeasure
from Decoder.Converter import DecimaltoBinary

def SLNNDecoder(nr_codeword, bit, snr_dB, model, model_pth, device):
    encoder = hamming74_encoder(device)

    bits_info = generator(nr_codeword, bit, device)  # Code Generator
    encoded_codeword = encoder(bits_info)  # Hamming(7,4) Encoder
    modulated_signal = bpsk_modulator(encoded_codeword)  # Modulate signal
    noised_signal = AWGN(modulated_signal, snr_dB, device)  # Add Noise

    practical_snr = NoiseMeasure(noised_signal, modulated_signal)

    # use SLNN model:
    model.eval()
    model.load_state_dict(torch.load(model_pth))

    SLNN_result = model(noised_signal)
    SLNN_decimal = torch.argmax(SLNN_result, dim=2)

    Decimal_Binary =DecimaltoBinary(device)
    SLNN_binary = Decimal_Binary(SLNN_decimal)


    return SLNN_binary, bits_info, practical_snr

def estimation(num, bit, SNR_opt_NN, SLNN_hidden_size, model_pth, result, i, device):
    N = num
    # Single-label Neural Network:
    input_size = 7
    output_size = torch.pow(bit, torch.tensor(2))

    model = SingleLabelNNDecoder(input_size, SLNN_hidden_size, output_size).to(device)
    SLNN_final, bits_info, snr_measure = SLNNDecoder(N, bit, SNR_opt_NN, model, model_pth, device)

    BLER_SLNN, error_num_SLNN = calculate_bler(SLNN_final, bits_info) # BER calculation

    if error_num_SLNN < 100:
        N += 1000000
        print(f"the code number is {N}")

    else:
        print(f"SLNN Hidden layer{i}: When SNR is {snr_measure} and signal number is {N}, error number is {error_num_SLNN} and BLER is {BLER_SLNN}")
        result[0, i] = BLER_SLNN

    return result


def main():
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.backends.cuda.is_available()
    #                 else torch.device("cpu")))
    device = torch.device("cpu")
    # device = torch.device("cuda")

    # Hyperparameters
    num = int(1e7)
    bit = 4
    SLNN_hidden_size = torch.arange(0, 101, 1)
    SNR_opt_NN = torch.tensor(8, dtype=torch.int, device=device)
    SNR_opt_NN = SNR_opt_NN + 10 * torch.log10(torch.tensor(4 / 7, dtype=torch.float)) # for SLNN article

    result_save = np.zeros((1, len(SLNN_hidden_size)))

    for i in range(0, len(SLNN_hidden_size)):
        save_pth = f"Result/Model/SLNN_CPU/SLNN_model_hiddenlayer{i}_BER0.pth"
        result_all = estimation(num, bit, SNR_opt_NN, SLNN_hidden_size[i], save_pth, result_save, i, device)
    directory_path = "Result/BLER"

    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"BLER_result_{current_time}.csv"
    full_csv_path = os.path.join(directory_path, csv_filename)
    np.savetxt(full_csv_path, result_all, delimiter=', ')


if __name__ == "__main__":
    main()