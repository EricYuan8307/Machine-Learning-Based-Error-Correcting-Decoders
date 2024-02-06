import torch
import numpy as np
import os
from datetime import datetime

from Encoder.Generator import generator
from Encoder.BPSK import bpsk_modulator
from Encoder.encoder import hamming74_encoder
from Decoder.NNDecoder import MultiLabelNNDecoder1, MultiLabelNNDecoder2
from Transmit.noise import AWGN
from Metric.ErrorRate import calculate_bler
from Transmit.NoiseMeasure import NoiseMeasure
from Decoder.Converter import MLNN_decision


# Calculate the Error number and BLER
def MLNNDecoder(nr_codeword, snr_dB, model, model_pth, device):
    encoder = hamming_encoder(device)

    bits_info = generator(nr_codeword, device)  # Code Generator
    encoded_codeword = encoder(bits_info)  # Hamming(7,4) Encoder
    modulated_signal = bpsk_modulator(encoded_codeword)  # Modulate signal
    noised_signal = AWGN(modulated_signal, snr_dB, device)  # Add Noise

    practical_snr = NoiseMeasure(noised_signal, modulated_signal)

    # use MLNN model:
    model.eval()
    model.load_state_dict(torch.load(model_pth))

    MLNN_final = model(noised_signal)

    return MLNN_final, bits_info, practical_snr

def estimation(num, SNR_opt_NN, MLNN_hidden_size, model_pth, result, device):
    N = num

    # Multi-label Neural Network with 1 hidden layer:
    for i in range(len(SNR_opt_NN)):
        snr_save = i / 2
        snr_dB = SNR_opt_NN[i]
        input_size = 7
        output_size = 4

        # model = MultiLabelNNDecoder1(input_size, MLNN_hidden_size, output_size).to(device)
        model = MultiLabelNNDecoder2(input_size, MLNN_hidden_size, output_size).to(device)
        MLNN_result, bits_info, snr_measure = MLNNDecoder(N, snr_dB, model, model_pth, device)
        MLNN_final = MLNN_decision(MLNN_result, device)

        BLER_MLNN, error_num_MLNN = calculate_bler(MLNN_final, bits_info)  # BER calculation

        if error_num_MLNN < 100:
            N += 1000000
            print(f"the code number is {N}")

        else:
            print(
                f"MLNN :hidden layer{MLNN_hidden_size}, When SNR is {snr_save} and signal number is {N}, error number is {error_num_MLNN} and BLER is {BLER_MLNN}")
            result[0, i] = BLER_MLNN

    return result


def main():
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.backends.cuda.is_available()
    #                 else torch.device("cpu")))
    device = torch.device("cpu")
    # device = torch.device("cuda")

    # Hyperparameters
    num = int(1e7)
    # MLNN_hidden_size = 100
    MLNN_hidden_size = [50, 50]
    SNR_opt_NN = torch.arange(0, 8.5, 0.5)
    SNR_opt_NN = SNR_opt_NN + 10 * torch.log10(torch.tensor(4 / 7, dtype=torch.float))  # for MLNN article

    model_save_pth = f"Result/Model/MLNN_CPU/MLNN_model_hiddenlayer{MLNN_hidden_size}_BER0.pth"

    result_save = np.zeros((1, len(SNR_opt_NN)))
    result_all = estimation(num, SNR_opt_NN, MLNN_hidden_size, model_save_pth, result_save, device)
    directory_path = "Result/BLER"

    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"BER_result_{current_time}.csv"
    full_csv_path = os.path.join(directory_path, csv_filename)
    np.savetxt(full_csv_path, result_all, delimiter=', ')


if __name__ == "__main__":
    main()