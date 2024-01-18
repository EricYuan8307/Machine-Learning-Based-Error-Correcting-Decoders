import torch
import numpy as np
import time
import os
from datetime import datetime

from Encoder.Generator import generator
from Encoder.BPSK import bpsk_modulator
from Encoder.Hamming74 import hamming_encoder
from Decoder.HardDecision import hard_decision
from Decoder.LDPC_BP import LDPCBeliefPropagation
from Decoder.likelihood import llr
from Decoder.NNDecoder import SingleLabelNNDecoder, MultiLabelNNDecoder
from Transmit.noise import AWGN
from Metric.ErrorRate import calculate_ber, calculate_bler
from Decoder.HammingDecoder import Hamming74decoder
from Decoder.MaximumLikelihood import HardDecisionML, SoftDecisionML
from Transmit.NoiseMeasure import NoiseMeasure


# Calculate the Error number and BER
def estimation(num, iter, SNR_opt_BPSK, SNR_opt_ML, SNR_opt_BP, SNR_opt_NN, MLNN_hidden_size, result, device):

    N = num

    # # De-Encoder, BPSK only
    # for i in range(len(SNR_opt_BPSK)):
    #     snr_dB =SNR_opt_BPSK[i]
    #
    #     for _ in range(10):
    #         BPSK_final, bits_info, snr_measure = UncodedBPSK(N, snr_dB, device)
    #
    #         BER_BPSK, error_num_BPSK= calculate_ber(BPSK_final, bits_info)
    #         if error_num_BPSK < 100:
    #             N += 2000000
    #             print(f"the code number is {N}")
    #
    #         else:
    #             print(f"BPSK: When SNR is {snr_measure} and signal number is {N}, error number is {error_num_BPSK} and BER is {BER_BPSK}")
    #             result[0, i] = BER_BPSK
    #             break


    # # Soft-Decision Maximum Likelihood
    # for i in range(len(SNR_opt_ML)):
    #     snr_dB = SNR_opt_ML[i]
    #
    #     for _ in range(10):
    #         SDML_final, bits_info, snr_measure = SoftDecisionMLP(N, snr_dB, device)
    #
    #         BER_SDML, error_num_SDML = calculate_ber(SDML_final, bits_info)
    #         # BER_SDML = calculate_bler(SDML_final, bits_info)
    #         if error_num_SDML < 100:
    #             N += 1000000
    #             print(f"the code number is {N}")
    #
    #         else:
    #             print(f"SD-ML: When SNR is {snr_measure} and signal number is {N}, error number is {error_num_SDML} and BER is {BER_SDML}")
    #             # print(f"SD-ML: When SNR is {snr_measure} and signal number is {N} and BLER is {BLER_SDML}")
    #             result[1, i] = BER_SDML
    #             break


    # # Hard-Decision Maximum Likelihood
    # for i in range(len(SNR_opt_ML)):
    #     snr_dB = SNR_opt_ML[i]
    #
    #     for _ in range(10):
    #         HDML_final, bits_info, snr_measure = HardDecisionMLP(N, snr_dB, device)
    #
    #         BER_HDML, error_num_HDML = calculate_ber(HDML_final, bits_info)
    #         if error_num_HDML < 100:
    #             N += 1000000
    #             print(f"the code number is {N}")
    #
    #         else:
    #             print(
    #                 f"HD-ML: When SNR is {snr_measure} and signal number is {N}, error number is {error_num_HDML} and BER is {BER_HDML}")
    #             result[2, i] = BER_HDML
    #             break


    # # Belief Propagation
    # for i in range(len(SNR_opt_BP)):
    #     snr_dB = SNR_opt_BP[i]
    #
    #     for _ in range(10):
    #         LDPC_final, bits_info, snr_measure = BeliefPropagation(N, snr_dB, iter, device)
    #
    #         BER_LDPC, error_num_LDPC = calculate_ber(LDPC_final, bits_info) # BER calculation
    #
    #         if error_num_LDPC < 100:
    #             N += 1000000
    #             print(f"the code number is {N}")
    #
    #         else:
    #             print(f"LDPC: When SNR is {snr_measure} and signal number is {N}, error number is {error_num_LDPC} and BER is {BER_LDPC}")
    #             result[3, i] = BER_LDPC
    #             break


    # Mluti-label Neural Network:
    for i in range(len(SNR_opt_NN)):
        snr_dB = SNR_opt_NN[i]
        input_size = 7
        output_size = 4

        model = MultiLabelNNDecoder(input_size, MLNN_hidden_size, output_size).to(device)
        MLNN_final, bits_info, snr_measure = MLNNDecoder(N, snr_dB, model, device)

        BER_MLNN, error_num_MLNN = calculate_ber(MLNN_final, bits_info) # BER calculation

        if error_num_MLNN < 100:
            N += 1000000
            print(f"the code number is {N}")

        else:
            print(f"MLNN: When SNR is {snr_measure} and signal number is {N}, error number is {error_num_MLNN} and BER is {BER_MLNN}")
            result[4, i] = BER_MLNN
            break



    return result


def UncodedBPSK(nr_codeword, snr_dB, device):
    bits_info = generator(nr_codeword, device)
    modulated_signal = bpsk_modulator(bits_info)
    noised_signal = AWGN(modulated_signal, snr_dB, device)

    BPSK_final = hard_decision(noised_signal, device)

    practical_snr = NoiseMeasure(noised_signal, modulated_signal)

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

def HardDecisionMLP(nr_codeword, snr_dB, device):
    encoder = hamming_encoder(device)
    HD_MaximumLikelihood = HardDecisionML(device)
    decoder = Hamming74decoder(device)

    # ML:
    bits_info = generator(nr_codeword, device)
    encoded_codeword = encoder(bits_info)
    modulated_signal = bpsk_modulator(encoded_codeword)
    noised_signal = AWGN(modulated_signal, snr_dB, device)

    HD_signal = hard_decision(noised_signal, device)
    HD_ML = HD_MaximumLikelihood(HD_signal)
    HDML_final = decoder(HD_ML)

    practical_snr = NoiseMeasure(noised_signal, modulated_signal)

    return HDML_final, bits_info, practical_snr

def BeliefPropagation(nr_codeword, snr_dB, iter, device):
    iter_start_time = time.time()

    encoder = hamming_encoder(device)
    ldpc_bp = LDPCBeliefPropagation(device)
    decoder = Hamming74decoder(device)

    bits_info = generator(nr_codeword, device)  # Code Generator
    encoded_codeword = encoder(bits_info)  # Hamming(7,4) Encoder
    modulated_signal = bpsk_modulator(encoded_codeword)  # Modulate signal
    noised_signal = AWGN(modulated_signal, snr_dB, device)  # Add Noise

    llr_output = llr(noised_signal, snr_dB)  # LLR
    BP_result = torch.zeros(llr_output.shape, device=device)

    practical_snr = NoiseMeasure(noised_signal, modulated_signal)

    for k in range(llr_output.shape[0]):
        start_time = time.time()

        BP = ldpc_bp(llr_output[k], iter)  # LDPC
        BP_result[k] = BP
        end_time = time.time()

        if k % 10000 == 0 and k > 0:
            elapsed_time = end_time - start_time
            print(f"Processed {k} iterations in {elapsed_time * 10000} seconds")

    iter_end_time = time.time()
    print(f"For {practical_snr}SNR, the Belief Propagation spend {iter_end_time - iter_start_time} seconds.")

    LDPC_HD = hard_decision(BP_result, device)  # Hard Decision
    LDPC_final = decoder(LDPC_HD)  # Decoder

    return LDPC_final, bits_info, practical_snr

def MLNNDecoder(nr_codeword, snr_dB, model, device):
    encoder = hamming_encoder(device)

    bits_info = generator(nr_codeword, device)  # Code Generator
    encoded_codeword = encoder(bits_info)  # Hamming(7,4) Encoder
    modulated_signal = bpsk_modulator(encoded_codeword)  # Modulate signal
    noised_signal = AWGN(modulated_signal, snr_dB, device)  # Add Noise

    practical_snr = NoiseMeasure(noised_signal, modulated_signal)

    # use MLNN model:
    model.eval()
    model.load_state_dict(torch.load(f"Result/Model/MLNN/MLNN_model_BER{snr_dB}.pth"))

    MLNN_final = model(noised_signal)

    return MLNN_final, bits_info, practical_snr


def main():
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.backends.cuda.is_available()
    #                 else torch.device("cpu")))
    device = torch.device("cpu")

    # Hpyer parameters
    num = int(1e8)
    iter = 5
    SNR_opt_BPSK = torch.arange(0, 10.5, 0.5)
    SNR_opt_ML = torch.arange(0, 9.5, 0.5)
    SNR_opt_BP = torch.arange(0, 9, 0.5)
    SNR_opt_NN = torch.tensor([4.0])
    SLNN_hidden_size = 100
    MLNN_hidden_size = 100

    result_save = np.zeros((7, len(SNR_opt_BPSK)))

    result_all = estimation(num, iter, SNR_opt_BPSK, SNR_opt_ML, SNR_opt_BP, SNR_opt_NN, MLNN_hidden_size, result_save, device)

    directory_path = "Result/BER"
    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Get the current timestamp as a string
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Construct the filename with the timestamp
    csv_filename = f"BER_result_{current_time}.csv"

    full_csv_path = os.path.join(directory_path, csv_filename)
    np.savetxt(full_csv_path, result_all, delimiter=',')


if __name__ == "__main__":
    main()