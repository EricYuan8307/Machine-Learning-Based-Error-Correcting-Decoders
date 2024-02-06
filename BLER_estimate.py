import torch
import numpy as np
import os
from datetime import datetime

from Encoder.Generator import generator
from Encoder.BPSK import bpsk_modulator
from Encoder.encoder import hamming74_encoder
from Decoder.HardDecision import hard_decision
from Decoder.NNDecoder import SingleLabelNNDecoder
from Transmit.noise import AWGN
from Metric.ErrorRate import calculate_bler
from Decoder.HammingDecoder import Hamming74decoder
from Decoder.MaximumLikelihood import SoftDecisionML
from Transmit.NoiseMeasure import NoiseMeasure, NoiseMeasure_BPSK
from Decoder.Converter import DecimaltoBinary


# Calculate the Error number and BLER
def UncodedBPSK(nr_codeword, snr_dB, device):
    bits_info = generator(nr_codeword, device)
    modulated_signal = bpsk_modulator(bits_info)
    noised_signal = AWGN(modulated_signal, snr_dB, device)

    BPSK_final = hard_decision(noised_signal, device)

    practical_snr = NoiseMeasure_BPSK(noised_signal, modulated_signal)

    return BPSK_final, bits_info, practical_snr

def SoftDecisionMLP(nr_codeword, snr_dB, device):
    encoder = hamming_encoder(device)
    SD_MaximumLikelihood = SoftDecisionML(device)
    decoder = Hamming74decoder(device)

    # ML:
    bits_info = generator(nr_codeword, device)
    encoded_codeword = encoder(bits_info)
    modulated_signal = bpsk_modulator(encoded_codeword)
    noised_signal = AWGN(modulated_signal, snr_dB, device)

    SD_ML = SD_MaximumLikelihood(noised_signal)
    HD_final = hard_decision(SD_ML, device)
    SDML_final = decoder(HD_final)

    practical_snr = NoiseMeasure(noised_signal, modulated_signal)

    return SDML_final, bits_info, practical_snr

def SLNNDecoder(nr_codeword, snr_dB, model, model_pth, device):
    encoder = hamming_encoder(device)

    bits_info = generator(nr_codeword, device)  # Code Generator
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

def estimation(num, SNR_opt_BPSK, SNR_opt_ML, SNR_opt_NN, SLNN_hidden_size, model_pth, result, device):
    N = num

    # De-Encoder, BPSK only
    for i in range(len(SNR_opt_BPSK)):
        snr_dB =SNR_opt_BPSK[i]

        for _ in range(10):
            BPSK_final, bits_info, snr_measure = UncodedBPSK(N, snr_dB, device)

            BLER_BPSK, error_num_BPSK= calculate_bler(BPSK_final, bits_info)
            if error_num_BPSK < 100:
                N += 2000000
                print(f"the code number is {N}")

            else:
                print(f"BPSK: When SNR is {snr_measure} and signal number is {N}, error number is {error_num_BPSK} and BER is {BLER_BPSK}")
                result[0, i] = BLER_BPSK
                break


    # Soft-Decision Maximum Likelihood
    for i in range(len(SNR_opt_ML)):
        snr_dB = SNR_opt_ML[i]

        # BLER
        for _ in range(10):
            SDML_final, bits_info, snr_measure = SoftDecisionMLP(N, snr_dB, device)

            BLER_SDML, block_error_num_SDML = calculate_bler(SDML_final, bits_info)
            if block_error_num_SDML < 100:
                N += 1000000
                print(f"the code number is {N}")

            else:
                print(f"SD-ML: When SNR is {snr_measure} and signal number is {N}, error number is {block_error_num_SDML} and BLER is {BLER_SDML}")
                result[1, i] = BLER_SDML
                break


    # Single-label Neural Network:
    for i in range(len(SNR_opt_NN)):
        snr_save = i / 2
        input_size = 7
        output_size = 16

        model = SingleLabelNNDecoder(input_size, SLNN_hidden_size, output_size).to(device)
        SLNN_final, bits_info, snr_measure = SLNNDecoder(N, SNR_opt_NN, model, model_pth, device)

        BLER_SLNN, error_num_SLNN = calculate_bler(SLNN_final, bits_info) # BER calculation

        if error_num_SLNN < 100:
            N += 1000000
            print(f"the code number is {N}")

        else:
            print(f"SLNN: When SNR is {snr_save} and signal number is {N}, error number is {error_num_SLNN} and BLER is {BLER_SLNN}")
            result[2, i] = BLER_SLNN


    return result


def main():
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.backends.cuda.is_available()
    #                 else torch.device("cpu")))
    # device = torch.device("cpu")
    device = torch.device("cuda")

    # Hyperparameters
    num = int(1e7)
    SLNN_hidden_size = 7
    SNR_opt_BPSK = torch.arange(0, 10.5, 0.5)
    SNR_opt_ML = torch.arange(0, 9.5, 0.5)
    SNR_opt_ML = SNR_opt_ML + 10 * torch.log10(torch.tensor(4 / 7, dtype=torch.float))  # for SLNN article
    SNR_opt_NN = torch.tensor(0.0, dtype=torch.float, device=device)
    SNR_opt_NN = SNR_opt_NN + 10 * torch.log10(torch.tensor(4 / 7, dtype=torch.float)) # for SLNN article

    result_save = np.zeros((7, len(SNR_opt_BPSK)))
    save_pth = "Result/Model/SLNN/SLNN_7/SLNN_model_BER0.0.pth"

    result_all = estimation(num, SNR_opt_BPSK, SNR_opt_ML, SNR_opt_NN, SLNN_hidden_size, save_pth, result_save, device)
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