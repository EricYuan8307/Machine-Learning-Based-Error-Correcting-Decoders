import torch
import numpy as np
import os

from Encode.Generator import generator
from Encode.Modulator import bpsk_modulator
from Decode.NNDecoder import SingleLabelNNDecoder1, MultiLabelNNDecoder_N
from Transmit.noise import AWGN
from Metric.ErrorRate import calculate_bler, calculate_ber
from Transmit.NoiseMeasure import NoiseMeasure, NoiseMeasure_MLNN
from Decode.Converter import DecimaltoBinary, MLNN_decision

from generating import all_codebook_NonML, SLNN_D2B_matrix
from Encode.Encoder import PCC_encoders

def SLNNDecoder(nr_codeword, method, bits, encoded, snr_dB, model, model_pth, batch_size, device):
    encoder_matrix, decoder_matrix = all_codebook_NonML(method, bits, encoded, device)
    SLNN_Matrix = SLNN_D2B_matrix(bits, device)

    encoder = PCC_encoders(encoder_matrix)
    convertor = DecimaltoBinary(SLNN_Matrix)

    bits_info = generator(nr_codeword, bits, device)  # Code Generator
    encoded_codeword = encoder(bits_info)  # Encoder
    modulated_signal = bpsk_modulator(encoded_codeword)  # Modulate signal
    noised_signal = AWGN(modulated_signal, snr_dB, device)  # Add Noise

    practical_snr = NoiseMeasure(noised_signal, modulated_signal, bits, encoded)

    # use SLNN model:
    model.eval()
    model.load_state_dict(torch.load(model_pth))

    # Process in batches
    num_batches = int(np.ceil(noised_signal.shape[0] / batch_size))
    SLNN_binary_batches = []
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, noised_signal.shape[0])
        noised_signal_batch = noised_signal[batch_start:batch_end]

        SLNN_result_batch = model(noised_signal_batch)
        SLNN_decimal_batch = torch.argmax(SLNN_result_batch, dim=2)
        SLNN_binary_batch = convertor(SLNN_decimal_batch)

        SLNN_binary_batches.append(SLNN_binary_batch)

        if i % 500 == 0 and i > 0:
            print(f"NN Decoder Batch: Processed batch: {i}/{num_batches}")

    # Concatenate the results
    SLNN_binary = torch.cat(SLNN_binary_batches, dim=0)


    return SLNN_binary, bits_info, practical_snr

def estimation_SLNN1(num, method, bits, encoded, NN_type, metric, SNR_opt_NN, NN_hidden_size, edge_deleted, model_pth, result, batch_size, order, t, device):
    N = num

    # Single-label Neural Network:
    condition_met = False
    iteration_count = 0  # Initialize iteration counter
    max_iterations = 50

    while not condition_met and iteration_count < max_iterations:
        iteration_count += 1

        if NN_type == "SLNN":
            output_size = torch.pow(torch.tensor(2), bits)
            model = SingleLabelNNDecoder1(encoded, NN_hidden_size, output_size).to(device)
            NN_final, bits_info, snr_measure = SLNNDecoder(N, method, bits, encoded, SNR_opt_NN, model, model_pth, batch_size, device)

        if metric == "BLER":
            error_rate, error_num = calculate_bler(NN_final, bits_info)
        elif metric == "BER":
            error_rate, error_num = calculate_ber(NN_final, bits_info)  # BER calculation

        if error_num < 100:
            N += int(1e7)
            print(f"the code number is {N}")

        else:
            print(
                f"{NN_type}, hiddenlayer{NN_hidden_size} - edgedeleted{edge_deleted}, order{order}: When SNR is {snr_measure} and signal number is {N}, error number is {error_num} and {metric} is {error_rate}")
            result[0, t] = error_rate
            condition_met = True

    return result

def MLNNDecoder(nr_codeword, method, bits, encoded, snr_dB, model, model_pth, batch_size, device):
    encoder_matrix, decoder_matrix = all_codebook_NonML(method, bits, encoded, device)

    encoder = PCC_encoders(encoder_matrix)

    bits_info = generator(nr_codeword, bits, device)  # Code Generator
    encoded_codeword = encoder(bits_info)  # Encoder
    modulated_signal = bpsk_modulator(encoded_codeword)  # Modulate signal
    noised_signal = AWGN(modulated_signal, snr_dB, device)  # Add Noise

    practical_snr = NoiseMeasure_MLNN(noised_signal, modulated_signal, bits, encoded)

    # use MLNN model:
    model.eval()
    model.load_state_dict(torch.load(model_pth))

    # Splitting the noised_signal into batches and processing each batch
    num_batches = int(np.ceil(noised_signal.shape[0] / batch_size))
    MLNN_final_batches = []
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, noised_signal.shape[0])
        noised_signal_batch = noised_signal[batch_start:batch_end]

        MLNN_result_batch = model(noised_signal_batch)
        MLNN_final_batches.append(MLNN_result_batch)

        if i % batch_size == 0 and i > 0:
            print(f"NN Decoder Batch: Processed {i} batches out of {num_batches}")

    # Concatenate the processed batches
    MLNN_final = torch.cat(MLNN_final_batches, dim=0)

    return MLNN_final, bits_info, practical_snr

def estimation_MLNN1(num, method, bits, encoded, NN_type, metric, SNR_opt_NN, NN_hidden_size, edge_deleted, model_pth, result, batch_size, order, i, device):
    N = num

    condition_met = False
    iteration_count = 0  # Initialize iteration counter
    max_iterations = 50

    while not condition_met and iteration_count < max_iterations:
        iteration_count += 1

        model = MultiLabelNNDecoder_N(encoded, NN_hidden_size, bits).to(device)
        NN_result, bits_info, snr_measure = MLNNDecoder(N, method, bits, encoded, SNR_opt_NN, model, model_pth, batch_size, device)
        NN_final = MLNN_decision(NN_result, device)

        if metric == "BLER":
            error_rate, error_num = calculate_bler(NN_final, bits_info)
        elif metric == "BER":
            error_rate, error_num = calculate_ber(NN_final, bits_info) # BER calculation

        if error_num < 100:
            N += int(1e7)
            print(f"the code number is {N}")

        else:
            print(
                f"{NN_type}, hiddenlayer{NN_hidden_size} - edgedeleted{edge_deleted}, order{order}: When SNR is {snr_measure} and signal number is {N}, error number is {error_num} and {metric} is {error_rate}")
            result[0, 0] = error_rate
            condition_met = True

    return result


def main():
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.cuda.is_available()
    #                 else torch.device("cpu")))
    device = torch.device("cpu")
    # device = torch.device("cuda")

    # Hyperparameters
    metrics = ["BER"]
    num = int(2e7)
    bits = 4
    encoded = 7
    encoding_method = "Hamming"
    NeuralNetwork_type = ["MLNN"]

    batch_size = int(1e4)
    SLNN_hidden_size1 = [24]
    MLNN_hidden_size1 = [16]
    edge_delete = 551
    # orders = torch.arange(1, 65, 1)
    orders = [1,2,3,6,8,10,11,12,13,14,16,17,18,21,23,25,26,29,31,33,35,36,38,39,40,41,43,44,46,48,49,52,53,55,56,59,60,62,64]
    SNR_opt_NN = torch.tensor(7, dtype=torch.float, device=device)
    # SNR_opt_NN = SNR_opt_NN + 10 * torch.log10(torch.tensor(bits / encoded, dtype=torch.float)) # for SLNN article

    result_save = np.zeros((1, len(orders)))

    for NN_type in NeuralNetwork_type:
        for metric in metrics:
            if NN_type == "SLNN":
                for i in range(len(SLNN_hidden_size1)):
                    model_pth = f"Result/Model/{encoding_method}{encoded}_{bits}/{NN_type}_{device}/{NN_type}_hiddenlayer{SLNN_hidden_size1[i]}.pth" # Normal
                    result_NN = estimation_SLNN1(num, encoding_method, bits, encoded, NN_type, metric, SNR_opt_NN, SLNN_hidden_size1[i], edge_delete, model_pth, result_save, batch_size, orders[i], i, device)
                    # directory_path = f"Result/{encoding_method}{encoded}_{bits}/{metric}" # Normal NN
                    directory_path = f"Result/{encoding_method}{encoded}_{bits}/{metric}_retrained"

                    # Create the directory if it doesn't exist
                    if not os.path.exists(directory_path):
                        os.makedirs(directory_path)

                    csv_filename = f"{metric}_{NN_type}_hiddenlayer{SLNN_hidden_size1[i]}.csv"
                    full_csv_path = os.path.join(directory_path, csv_filename)
                    np.savetxt(full_csv_path, result_NN, delimiter=', ')

            if NN_type == "MLNN":
                for i in range(len(MLNN_hidden_size1)):
                    for order in orders:
                        model_pth = f"Result/Model/{encoding_method}{encoded}_{bits}/{MLNN_hidden_size1[i]}_ft_{device}/{NN_type}_deleted{order}.pth"
                        result_NN = estimation_MLNN1(num, encoding_method, bits, encoded, NN_type, metric, SNR_opt_NN,
                                                     MLNN_hidden_size1[i], edge_delete, model_pth, result_save, batch_size, order, i, device)

                        directory_path = f"Result/{encoding_method}{encoded}_{bits}/{metric}"

                        # Create the directory if it doesn't exist
                        if not os.path.exists(directory_path):
                            os.makedirs(directory_path)

                        csv_filename = f"{metric}_{NN_type}_deleted{order}.csv"
                        full_csv_path = os.path.join(directory_path, csv_filename)
                        np.savetxt(full_csv_path, result_NN, delimiter=', ')


if __name__ == "__main__":
    main()