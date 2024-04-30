import torch
import numpy as np
import os

from Encode.Generator import generator
from Encode.Modulator import bpsk_modulator
from Decode.NNDecoder import SingleLabelNNDecoder1, SingleLabelNNDecoder2, MultiLabelNNDecoder1, MultiLabelNNDecoder2, MultiLabelNNDecoder3, MultiLabelNNDecoder_N
from Transmit.noise import AWGN
from Metric.ErrorRate import calculate_ber, calculate_bler
from Transmit.NoiseMeasure import NoiseMeasure
from Decode.Converter import DecimaltoBinary
from generating import all_codebook_NonML, SLNN_D2B_matrix
from Encode.Encoder import PCC_encoders
from Decode.Converter import MLNN_decision


# Calculate the Error number and BLER
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

def MLNNDecoder(nr_codeword, method, bits, encoded, snr_dB, model, model_pth, batch_size, device):
    encoder_matrix, decoder_matrix = all_codebook_NonML(method, bits, encoded, device)

    encoder = PCC_encoders(encoder_matrix)

    bits_info = generator(nr_codeword, bits, device)  # Code Generator
    encoded_codeword = encoder(bits_info)  # Encoder
    modulated_signal = bpsk_modulator(encoded_codeword)  # Modulate signal
    noised_signal = AWGN(modulated_signal, snr_dB, device)  # Add Noise

    practical_snr = NoiseMeasure(noised_signal, modulated_signal, bits, encoded)

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

def estimation_SLNN1(num, method, bits, encoded, NN_type, metric, SNR_opt_NN, NN_hidden_size, model_pth, result, batch_size, device):
    N = num

    # Single-label Neural Network:
    for i in range(len(SNR_opt_NN)):
        condition_met = False
        iteration_count = 0  # Initialize iteration counter
        max_iterations = 50

        while not condition_met and iteration_count < max_iterations:
            iteration_count += 1

            if NN_type == "SLNN":
                output_size = torch.pow(torch.tensor(2), bits)
                model = SingleLabelNNDecoder1(encoded, NN_hidden_size, output_size).to(device)
                NN_final, bits_info, snr_measure = SLNNDecoder(N, method, bits, encoded, SNR_opt_NN[i], model, model_pth, batch_size, device)

            if metric == "BLER":
                error_rate, error_num = calculate_bler(NN_final, bits_info)
            elif metric == "BER":
                error_rate, error_num = calculate_ber(NN_final, bits_info)  # BER calculation

            if error_num < 100:
                N += int(1e7)
                print(f"the code number is {N}")

            else:
                print(
                    f"{NN_type}, hiddenlayer{NN_hidden_size}: When SNR is {snr_measure} and signal number is {N}, error number is {error_num} and {metric} is {error_rate}")
                result[0, i] = error_rate
                condition_met = True

    return result

def estimation_MLNN1(num, method, bits, encoded, NN_type, metric, SNR_opt_NN, NN_hidden_size, model_pth, result, batch_size, device):
    N = num

    # Single-label Neural Network:
    for i in range(len(SNR_opt_NN)):
        condition_met = False
        iteration_count = 0  # Initialize iteration counter
        max_iterations = 50

        while not condition_met and iteration_count < max_iterations:
            iteration_count += 1

            model = MultiLabelNNDecoder_N(encoded, NN_hidden_size, bits).to(device)
            NN_result, bits_info, snr_measure = MLNNDecoder(N, method, bits, encoded, SNR_opt_NN[i], model, model_pth, batch_size, device)
            NN_final = MLNN_decision(NN_result, device)

            if metric == "BLER":
                error_rate, error_num = calculate_bler(NN_final, bits_info)
            elif metric == "BER":
                error_rate, error_num = calculate_ber(NN_final, bits_info) # BER calculation

            if error_num < 100:
                N += int(1e7)
                print(f"the code number is {N}")

            else:
                print(f"{NN_type}, hiddenlayer{NN_hidden_size}: When SNR is {snr_measure} and signal number is {N}, error number is {error_num} and {metric} is {error_rate}")
                result[0, i] = error_rate
                condition_met = True

    return result

def estimation_NN(num, method, bits, encoded, NN_type, metric, SNR_opt_NN, NN_hidden_size, model_pth, result, batch_size, device):
    N = num
    hiddenlayer_num = len(NN_hidden_size)

    # Single-label Neural Network:
    for i in range(len(SNR_opt_NN)):
        condition_met = False
        iteration_count = 0  # Initialize iteration counter
        max_iterations = 50

        while not condition_met and iteration_count < max_iterations:
            iteration_count += 1

            if NN_type == "SLNN":
                output_size = torch.pow(torch.tensor(2), bits)
                if hiddenlayer_num == 2:
                    model = SingleLabelNNDecoder2(encoded, NN_hidden_size, output_size).to(device)
                NN_final, bits_info, snr_measure = SLNNDecoder(N, method, bits, encoded, SNR_opt_NN[i], model, model_pth, batch_size, device)

            elif NN_type == "MLNN":
                if hiddenlayer_num == 2:
                    model = MultiLabelNNDecoder2(encoded, NN_hidden_size, bits).to(device)
                elif hiddenlayer_num == 3:
                    model = MultiLabelNNDecoder3(encoded, NN_hidden_size, bits).to(device)
                NN_result, bits_info, snr_measure = MLNNDecoder(N, method, bits, encoded, SNR_opt_NN[i], model, model_pth, batch_size, device)
                NN_final = MLNN_decision(NN_result, device)

            if metric == "BLER":
                error_rate, error_num = calculate_bler(NN_final, bits_info)
            elif metric == "BER":
                error_rate, error_num = calculate_ber(NN_final, bits_info) # BER calculation

            if error_num < 100:
                N += int(1e7)
                print(f"the code number is {N}")

            else:
                print(f"{NN_type}, hiddenlayer{NN_hidden_size}: When SNR is {snr_measure} and signal number is {N}, error number is {error_num} and {metric} is {error_rate}")
                result[0, i] = error_rate
                condition_met = True

    return result


def main():
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.cuda.is_available()
    #                 else torch.device("cpu")))
    device = torch.device("cpu")
    # device = torch.device("cuda")

    # Hyperparameters
    metrics = ["BER"] # ["BER", "BLER"]
    nr_codeword = int(1e7)
    bits = 4
    encoded = 7
    encoding_method = "Hamming"  # "Hamming", "Parity", "BCH"
    NeuralNetwork_type = ["MLNN"] # ["SLNN", "MLNN"]
    batch_size = int(1e4)
    # SLNN_hidden_size1 = [24] # [20, 21, 22, 23, 24, 25, 26, 27, 28]
    # SLNN_hidden_size2 = [[25, 25], [100, 20], [20, 100], [100, 25], [25, 100]]
    MLNN_hidden_size1 = [16]
    MLNN_hidden_size2 = [[50,50], [100, 100]]

    SNR_opt_NN = torch.arange(0, 8.5, 0.5).to(device)
    SNR_opt_NN = SNR_opt_NN + 10 * torch.log10(torch.tensor(bits / encoded, dtype=torch.float))
    result_save = np.zeros((1, len(SNR_opt_NN)))

    # For trained and deleted model
    for NN_type in NeuralNetwork_type:
        for metric in metrics:
            # if NN_type == "SLNN":
                # for i in range(len(SLNN_hidden_size1)):
                #     model_pth = f"Result/Model/{encoding_method}{encoded}_{bits}/{NN_type}_{device}/{NN_type}_hiddenlayer{SLNN_hidden_size1[i]}.pth" # Normal
                #     result_NN = estimation_SLNN1(nr_codeword, encoding_method, bits, encoded, NN_type, metric, SNR_opt_NN, SLNN_hidden_size1[i], model_pth, result_save, batch_size, device)
                #
                #     # directory_path = f"Result/{encoding_method}{encoded}_{bits}/{metric}" # Normal NN
                #     directory_path = f"Result/{encoding_method}{encoded}_{bits}/{metric}_retrained"
                #
                #     # Create the directory if it doesn't exist
                #     if not os.path.exists(directory_path):
                #         os.makedirs(directory_path)
                #
                #     csv_filename = f"{metric}_{NN_type}_hiddenlayer{SLNN_hidden_size1[i]}.csv"
                #     full_csv_path = os.path.join(directory_path, csv_filename)
                #     np.savetxt(full_csv_path, result_NN, delimiter=', ')

            if NN_type == "MLNN":
                for MLNN_hidden_size in MLNN_hidden_size1:
                    model_pth = f"Result/Model/{encoding_method}{encoded}_{bits}/{NN_type}_{device}/{NN_type}_hiddenlayer{MLNN_hidden_size}.pth"
                    result_NN = estimation_MLNN1(nr_codeword, encoding_method, bits, encoded, NN_type, metric, SNR_opt_NN, MLNN_hidden_size, model_pth, result_save, batch_size, device)

                    directory_path = f"Result/{encoding_method}{encoded}_{bits}/{metric}"

                    # Create the directory if it doesn't exist
                    if not os.path.exists(directory_path):
                        os.makedirs(directory_path)

                    csv_filename = f"{metric}_{NN_type}_hiddenlayer{MLNN_hidden_size}.csv"
                    full_csv_path = os.path.join(directory_path, csv_filename)
                    np.savetxt(full_csv_path, result_NN, delimiter=', ')

    # for NN_type in NeuralNetwork_type:
    #     for metric in metrics:
    #         if NN_type == "SLNN":
    #             for i in range(len(SLNN_hidden_size1)):
    #                 # model_pth = f"Result/Model/{encoding_method}{encoded}_{bits}/{NN_type}_{device}/{NN_type}_hiddenlayer{SLNN_hidden_size1[i]}.pth" # Normal
    #                 model_pth = f"Result/Model/{encoding_method}{encoded}_{bits}/Parity_26_ft_{device}/{NN_type}_edgedeleted{edge_deleted}_trained.pth"  # Normal NN
    #                 result_NN = estimation_SLNN1(nr_codeword, encoding_method, bits, encoded, NN_type, metric, SNR_opt_NN, SLNN_hidden_size1[i], model_pth, result_save, batch_size, device)
    #
    #                 # directory_path = f"Result/{encoding_method}{encoded}_{bits}/{metric}" # Normal NN
    #                 directory_path = f"Result/{encoding_method}{encoded}_{bits}/{metric}_retrained"
    #
    #                 # Create the directory if it doesn't exist
    #                 if not os.path.exists(directory_path):
    #                     os.makedirs(directory_path)
    #
    #                 csv_filename = f"{metric}_{NN_type}_hiddenlayer{SLNN_hidden_size1[i]}.csv"
    #                 full_csv_path = os.path.join(directory_path, csv_filename)
    #                 np.savetxt(full_csv_path, result_NN, delimiter=', ')
    #
    #             for j in range(len(SLNN_hidden_size2)):
    #                 model_pth = f"Result/Model/{encoding_method}{encoded}_{bits}/{NN_type}_{device}/{NN_type}_hiddenlayer{SLNN_hidden_size2[j]}.pth"
    #                 result_NN = estimation_NN(nr_codeword, encoding_method, bits, encoded, NN_type, metric, SNR_opt_NN, SLNN_hidden_size2[j], model_pth, result_save, batch_size, device)
    #
    #                 directory_path = f"Result/{encoding_method}{encoded}_{bits}/{metric}"
    #
    #                 # Create the directory if it doesn't exist
    #                 if not os.path.exists(directory_path):
    #                     os.makedirs(directory_path)
    #
    #                 csv_filename = f"{metric}_{NN_type}_hiddenlayer{SLNN_hidden_size2[j]}.csv"
    #                 full_csv_path = os.path.join(directory_path, csv_filename)
    #                 np.savetxt(full_csv_path, result_NN, delimiter=', ')
    #
    #         elif NN_type == "MLNN":
    #             for k in range(len(MLNN_hidden_size2)):
    #                 model_pth = f"Result/Model/{encoding_method}{encoded}_{bits}/{NN_type}_{device}/{NN_type}_hiddenlayer{MLNN_hidden_size2[k]}.pth"
    #                 result_NN = estimation_NN(nr_codeword, encoding_method, bits, encoded, NN_type, metric, SNR_opt_NN, MLNN_hidden_size2[k], model_pth, result_save, batch_size, device)
    #
    #                 directory_path = f"Result/{encoding_method}{encoded}_{bits}/{metric}"
    #
    #                 # Create the directory if it doesn't exist
    #                 if not os.path.exists(directory_path):
    #                     os.makedirs(directory_path)
    #
    #                 csv_filename = f"{metric}_{NN_type}_hiddenlayer{MLNN_hidden_size2[k]}.csv"
    #                 full_csv_path = os.path.join(directory_path, csv_filename)
    #                 np.savetxt(full_csv_path, result_NN, delimiter=', ')


if __name__ == "__main__":
    main()