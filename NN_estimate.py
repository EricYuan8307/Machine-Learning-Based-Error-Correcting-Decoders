import torch
import numpy as np
import os
from datetime import datetime

from Encode.Generator import generator
from Encode.Modulator import bpsk_modulator
from Decode.NNDecoder import SingleLabelNNDecoder1, SingleLabelNNDecoder2, MultiLabelNNDecoder1, MultiLabelNNDecoder2, MultiLabelNNDecoder3
from Transmit.noise import AWGN
from Metric.ErrorRate import calculate_ber, calculate_bler
from Transmit.NoiseMeasure import NoiseMeasure
from Decode.Converter import DecimaltoBinary
from generating import all_codebook_NonML, SLNN_D2B_matrix
from Encode.Encoder import PCC_encoders
from Decode.Converter import MLNN_decision



# Calculate the Error number and BLER
def SLNNDecoder(nr_codeword, method, bits, encoded, snr_dB, model, model_pth, device):
    encoder_matrix, decoder_matrix = all_codebook_NonML(method, bits, encoded, device)
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

def MLNNDecoder(nr_codeword, method, bits, encoded, snr_dB, model, model_pth, device):
    encoder_matrix, decoder_matrix = all_codebook_NonML(method, bits, encoded, device)

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

def estimation_SLNN1(num, method, bits, encoded, NN_type, metric, SNR_opt_NN, NN_hidden_size, model_pth, result, device):
    N = num

    # Single-label Neural Network:
    for i in range(len(SNR_opt_NN)):
        snr_save = i / 2

        if NN_type == "SLNN":
            output_size = torch.pow(torch.tensor(2), bits)
            model = SingleLabelNNDecoder1(encoded, NN_hidden_size, output_size).to(device)
            NN_final, bits_info, snr_measure = SLNNDecoder(N, method, bits, encoded, SNR_opt_NN[i], model, model_pth, device)

        if metric == "BLER":
            error_rate, error_num = calculate_bler(NN_final, bits_info)
        elif metric == "BER":
            error_rate, error_num = calculate_ber(NN_final, bits_info)  # BER calculation

        if error_num < 100:
            N += 1000000
            print(f"the code number is {N}")

        else:
            print(
                f"{NN_type}, hiddenlayer{NN_hidden_size}: When SNR is {snr_save} and signal number is {N}, error number is {error_num} and {metric} is {error_rate}")
            result[0, i] = error_rate

    return result

def estimation_NN(num, method, bits, encoded, NN_type, metric, SNR_opt_NN, NN_hidden_size, model_pth, result, device):
    N = num
    hiddenlayer_num = len(NN_hidden_size)

    # Single-label Neural Network:
    for i in range(len(SNR_opt_NN)):
        snr_save = i / 2

        if NN_type == "SLNN":
            output_size = torch.pow(torch.tensor(2), bits)
            if hiddenlayer_num == 2:
                model = SingleLabelNNDecoder2(encoded, NN_hidden_size, output_size).to(device)
            NN_final, bits_info, snr_measure = SLNNDecoder(N, method, bits, encoded, SNR_opt_NN[i], model, model_pth, device)

        elif NN_type == "MLNN":
            if hiddenlayer_num == 1:
                model = MultiLabelNNDecoder1(encoded, NN_hidden_size, bits).to(device)
            elif hiddenlayer_num == 2:
                model = MultiLabelNNDecoder2(encoded, NN_hidden_size, bits).to(device)
            elif hiddenlayer_num == 3:
                model = MultiLabelNNDecoder3(encoded, NN_hidden_size, bits).to(device)
            NN_result, bits_info, snr_measure = MLNNDecoder(N, method, bits, encoded, snr_save, model, model_pth, device)
            NN_final = MLNN_decision(NN_result, device)

        if metric == "BLER":
            error_rate, error_num = calculate_bler(NN_final, bits_info)
        elif metric == "BER":
            error_rate, error_num = calculate_ber(NN_final, bits_info) # BER calculation

        if error_num < 100:
            N += 1000000
            print(f"the code number is {N}")

        else:
            print(f"{NN_type}, hiddenlayer{NN_hidden_size}: When SNR is {snr_save} and signal number is {N}, error number is {error_num} and {metric} is {error_rate}")
            result[0, i] = error_rate

    return result


def main():
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.cuda.is_available()
    #                 else torch.device("cpu")))
    # device = torch.device("cpu")
    device = torch.device("cuda")

    # Hyperparameters
    metrics = ["BER", "BLER"]
    nr_codeword = int(1e7)
    bits = 10
    encoded = 26
    encoding_method = "Parity"  # "Hamming", "Parity", "BCH"
    NeuralNetwork_type = ["SLNN"] # ["SLNN", "MLNN"]
    SLNN_hidden_size1 = [24, 25, 26, 27, 28]
    SLNN_hidden_size2 = [[25, 25], [100, 20], [20, 100], [100, 25], [25, 100]]
    MLNN_hidden_size = [[1000, 500], [2000, 1000], [2000, 1000, 500]]

    SNR_opt_NN = torch.arange(0, 8.5, 0.5).to(device)
    SNR_opt_NN = SNR_opt_NN + 10 * torch.log10(torch.tensor(bits / encoded, dtype=torch.float))
    result_save = np.zeros((1, len(SNR_opt_NN)))

    for NN_type in NeuralNetwork_type:
        for metric in metrics:
            if NN_type == "SLNN":
                for i in range(len(SLNN_hidden_size1)):
                    model_pth = f"Result/Model/{encoding_method}{encoded}_{bits}/{NN_type}_{device}/{NN_type}_hiddenlayer{SLNN_hidden_size1[i]}.pth"
                    result_NN = estimation_SLNN1(nr_codeword, encoding_method, bits, encoded, NN_type, metric, SNR_opt_NN, SLNN_hidden_size1[i], model_pth, result_save, device)

                    directory_path = f"Result/{encoding_method}{encoded}_{bits}/{metric}"

                    # Create the directory if it doesn't exist
                    if not os.path.exists(directory_path):
                        os.makedirs(directory_path)

                    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    csv_filename = f"{metric}_result_{current_time}.csv"
                    full_csv_path = os.path.join(directory_path, csv_filename)
                    np.savetxt(full_csv_path, result_NN, delimiter=', ')

                for j in range(len(SLNN_hidden_size2)):
                    model_pth = f"Result/Model/{encoding_method}{encoded}_{bits}/{NN_type}_{device}/{NN_type}_hiddenlayer{SLNN_hidden_size2[j]}.pth"
                    result_NN = estimation_NN(nr_codeword, encoding_method, bits, encoded, NN_type, metric, SNR_opt_NN, SLNN_hidden_size2[j], model_pth, result_save, device)

                    directory_path = f"Result/{encoding_method}{encoded}_{bits}/{metric}"

                    # Create the directory if it doesn't exist
                    if not os.path.exists(directory_path):
                        os.makedirs(directory_path)

                    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    csv_filename = f"{metric}_result_{current_time}.csv"
                    full_csv_path = os.path.join(directory_path, csv_filename)
                    np.savetxt(full_csv_path, result_NN, delimiter=', ')

            elif NN_type == "MLNN":
                for k in range(len(MLNN_hidden_size)):
                    model_pth = f"Result/Model/{encoding_method}{encoded}_{bits}/{NN_type}_{device}/{NN_type}_hiddenlayer{MLNN_hidden_size[k]}.pth"
                    result_NN = estimation_NN(nr_codeword, encoding_method, bits, encoded, NN_type, metric, SNR_opt_NN, MLNN_hidden_size[k], model_pth, result_save, device)

                    directory_path = f"Result/{encoding_method}{encoded}_{bits}/{metric}"

                    # Create the directory if it doesn't exist
                    if not os.path.exists(directory_path):
                        os.makedirs(directory_path)

                    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    csv_filename = f"{metric}_result_{current_time}.csv"
                    full_csv_path = os.path.join(directory_path, csv_filename)
                    np.savetxt(full_csv_path, result_NN, delimiter=', ')


if __name__ == "__main__":
    main()