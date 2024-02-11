import torch
import numpy as np
import os
from datetime import datetime

from Encode.Generator import generator
from Encode.Modulator import bpsk_modulator
from Decode.NNDecoder import MultiLabelNNDecoder1, MultiLabelNNDecoder2
from Transmit.noise import AWGN
from Metric.ErrorRate import calculate_ber
from Transmit.NoiseMeasure import NoiseMeasure
from Decode.Converter import MLNN_decision
from generating import all_codebook
from Encode.Encoder import PCC_encoders


# Calculate the Error number and BER
def MLNNDecoder(nr_codeword, bits, encoded, snr_dB, model, model_pth, device):
    encoder_matrix, decoder_matrix, SoftDecisionMLMatrix = all_codebook(bits, encoded, device)
    encoder = PCC_encoders(encoder_matrix)

    bits_info = generator(nr_codeword, bits, device)  # Code Generator
    encoded_codeword = encoder(bits_info)  # Hamming(7,4) Encoder
    modulated_signal = bpsk_modulator(encoded_codeword)  # Modulate signal
    noised_signal = AWGN(modulated_signal, snr_dB, device)  # Add Noise

    practical_snr = NoiseMeasure(noised_signal, modulated_signal, bits, encoded)

    # use MLNN model:
    model.eval()
    model.load_state_dict(torch.load(model_pth))

    MLNN_final = model(noised_signal)

    return MLNN_final, bits_info, practical_snr

def estimation_MLNN1(num, bits, encoded, SNR_opt_NN, MLNN_hidden_size, model_pth, result, device):
    N = num

    # Multi-label Neural Network:
    for i in range(len(SNR_opt_NN)):
        snr_save = i/2
        snr_dB = SNR_opt_NN[i]
        input_size = 7

        model = MultiLabelNNDecoder1(input_size, MLNN_hidden_size, bits).to(device)
        MLNN_result, bits_info, snr_measure = MLNNDecoder(N, bits, encoded, snr_dB, model, model_pth, device)
        MLNN_final = MLNN_decision(MLNN_result, device)

        BER_MLNN, error_num_MLNN = calculate_ber(MLNN_final, bits_info) # BER calculation

        if error_num_MLNN < 100:
            N += 1000000
            print(f"the code number is {N}")

        else:
            print(f"MLNN: When SNR is {snr_save} and signal number is {N}, error number is {error_num_MLNN} and BER is {BER_MLNN}")
            result[0, i] = BER_MLNN

    return result

def estimation_MLNN2(num, bits, encoded, SNR_opt_NN, MLNN_hidden_size, model_pth, result, device):
    N = num

    # Multi-label Neural Network:
    for i in range(len(SNR_opt_NN)):
        snr_save = i/2
        snr_dB = SNR_opt_NN[i]
        input_size = 7

        model = MultiLabelNNDecoder2(input_size, MLNN_hidden_size, bits).to(device)
        MLNN_result, bits_info, snr_measure = MLNNDecoder(N, bits, encoded, snr_dB, model, model_pth, device)
        MLNN_final = MLNN_decision(MLNN_result, device)

        BER_MLNN, error_num_MLNN = calculate_ber(MLNN_final, bits_info) # BER calculation

        if error_num_MLNN < 100:
            N += 1000000
            print(f"the code number is {N}")

        else:
            print(f"MLNN: When SNR is {snr_save} and signal number is {N}, error number is {error_num_MLNN} and BER is {BER_MLNN}")
            result[0, i] = BER_MLNN

    return result

def main():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else (torch.device("cuda") if torch.backends.cuda.is_available()
                    else torch.device("cpu")))
    # device = torch.device("cpu")
    # device = torch.device("cuda")

    # Hyperparameters
    num = int(1e7)
    bits = 4
    encoded = 7
    MLNN_hidden_size = 100
    MLNN2_hidden_size = [[50,50], [100, 100]]
    SNR_opt_NN = torch.arange(0, 8.5, 0.5)
    SNR_opt_NN = SNR_opt_NN + 10 * torch.log10(torch.tensor(bits / encoded, dtype=torch.float)) # for MLNN article


    model_save_pth100 = "Result/Model/MLNN_CPU/MLNN_model_hiddenlayer100_BER0.pth"
    model_save_pth50_50 = "Result/Model/MLNN_CPU/MLNN_model_hiddenlayer[50, 50]_BER0.pth"
    model_save_pth100_100 = "Result/Model/MLNN_CPU/MLNN_model_hiddenlayer[100, 100]_BER0.pth"

    result_save = np.zeros((1, len(SNR_opt_NN)))
    result_MLNN100 = estimation_MLNN1(num, bits, encoded, SNR_opt_NN, MLNN_hidden_size, model_save_pth100, result_save, device)
    result_MLNN50_50 = estimation_MLNN2(num, bits, encoded, SNR_opt_NN, MLNN2_hidden_size[0], model_save_pth50_50, result_save, device)
    result_MLNN100_100 = estimation_MLNN2(num, bits, encoded, SNR_opt_NN, MLNN2_hidden_size[1], model_save_pth100_100, result_save, device)


    result_all = np.vstack([
                            result_MLNN100,
                            result_MLNN50_50,
                            result_MLNN100_100
                            ])


    directory_path = "Result/BER"

    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"BER_result_{current_time}.csv"
    full_csv_path = os.path.join(directory_path, csv_filename)
    np.savetxt(full_csv_path, result_all, delimiter=', ')


if __name__ == "__main__":
    main()