import torch
import numpy as np
import os
from datetime import datetime

from Encode.Generator import generator
from Encode.Modulator import bpsk_modulator
from Decode.NNDecoder import SingleLabelNNDecoder
from Transmit.noise import AWGN
from Metric.ErrorRate import calculate_ber
from Transmit.NoiseMeasure import NoiseMeasure
from Decode.Converter import DecimaltoBinary
from generating import all_codebook, SLNN_D2B_matrix
from Encode.Encoder import PCC_encoders


# Calculate the Error number and BLER
def SLNNDecoder(nr_codeword, bits, encoded, snr_dB, model, model_pth, device):
    encoder_matrix, decoder_matrix, SoftDecisionMLMatrix = all_codebook(bits, encoded, device)
    SLNN_Matrix = SLNN_D2B_matrix(bits, encoded, device)


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

def estimation_SLNN(num, bits, encoded, SNR_opt_NN, SLNN_hidden_size, model_pth, result, device):
    N = num

    # Single-label Neural Network:
    for i in range(len(SNR_opt_NN)):
        snr_save = i / 2
        input_size = 7
        output_size = torch.pow(torch.tensor(2), bits)

        model = SingleLabelNNDecoder(input_size, SLNN_hidden_size, output_size).to(device)
        SLNN_final, bits_info, snr_measure = SLNNDecoder(N, bits, encoded, SNR_opt_NN[i], model, model_pth, device)

        BER_SLNN, error_num_SLNN = calculate_ber(SLNN_final, bits_info) # BER calculation

        if error_num_SLNN < 100:
            N += 1000000
            print(f"the code number is {N}")

        else:
            print(f"SLNN: When SNR is {snr_save} and signal number is {N}, error number is {error_num_SLNN} and BLER is {BER_SLNN}")
            result[0, i] = BER_SLNN

    return result


def main():
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.backends.cuda.is_available()
    #                 else torch.device("cpu")))
    device = torch.device("cpu")
    # device = torch.device("cuda")

    # Hyperparameters
    num = int(1e7)
    bits = 4
    encoded = 7
    SLNN_hidden_size = 7
    SNR_opt_NN = torch.arange(0, 8.5, 0.5).to(device)
    SNR_opt_NN = SNR_opt_NN + 10 * torch.log10(torch.tensor(bits / encoded, dtype=torch.float)) # for SLNN article

    save_pth = "Result/Model/SLNN_CPU/SLNN_model_hiddenlayer7_BER0.pth"

    result_save = np.zeros((1, len(SNR_opt_NN)))
    result_SLNN = estimation_SLNN(num, bits, encoded, SNR_opt_NN, SLNN_hidden_size, save_pth, result_save, device)

    directory_path = "Result/BLER"

    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"BLER_result_{current_time}.csv"
    full_csv_path = os.path.join(directory_path, csv_filename)
    np.savetxt(full_csv_path, result_SLNN, delimiter=', ')


if __name__ == "__main__":
    main()