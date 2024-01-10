import torch
import numpy as np
import time
import os
import matplotlib.pyplot as plt

from Encoder.BPSK import bpsk_modulator
from Encoder.Hamming74 import hamming_encoder
from Decoder.HardDecision import hard_decision
from Decoder.LDPC_BP import LDPCBeliefPropagation
from Decoder.likelihood import llr
from Transmit.noise import AWGN
from Estimation.BitErrorRate import calculate_ber
from Decoder.HammingDecoder import Hamming74decoder
from Decoder.MaximumLikelihood import HardDecisionML, SoftDecisionML
from Transmit.NoiseMeasure import NoiseMeasure


# Code Generation
def generator(nr_codewords, device):
    bits = torch.randint(0, 2, size=(nr_codewords, 1, 4), dtype=torch.int, device=device)
    # bits = torch.tensor([[[1, 1, 0, 1]], [[0, 1, 0, 0,]]], dtype=torch.int, device=device)

    return bits


# Calculate the Error number and BER
def main():
    SNR_opt_BPSK = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
    SNR_opt_ML = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5]
    SNR_opt_BP = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9]


    result = np.zeros((4, len(SNR_opt_BPSK)))
    N = num

    # De-Encoder, BPSK only
    for i in range(len(SNR_opt_BPSK)):
        snr_dB =SNR_opt_BPSK[i]

        for _ in range(10):
            BPSK_final, bits_info, snr_measure = UncodedBPSK(N, snr_dB, device)

            BER_BPSK, error_num_BPSK= calculate_ber(BPSK_final, bits_info)
            if error_num_BPSK < 100:
                N += 2000000
                print(f"the code number is {N}")

            else:
                print(f"BPSK: When SNR is {snr_measure} and signal number is {N}, error number is {error_num_BPSK} and BER is {BER_BPSK}")
                result[0, i] = BER_BPSK
                break


    # Soft-Decision Maximum Likelihood
    for i in range(len(SNR_opt_ML)):
        snr_dB = SNR_opt_ML[i]

        for _ in range(10):
            SDML_final, bits_info, snr_measure = SoftDecisionMLP(N, snr_dB, device)

            BER_SDML, error_num_SDML = calculate_ber(SDML_final, bits_info)
            if error_num_SDML < 100:
                N += 1000000
                print(f"the code number is {N}")

            else:
                print(
                    f"SD-ML: When SNR is {snr_measure} and signal number is {N}, error number is {error_num_SDML} and BER is {BER_SDML}")
                result[1, i] = BER_SDML
                break


    # Hard-Decision Maximum Likelihood
    for i in range(len(SNR_opt_ML)):
        snr_dB = SNR_opt_ML[i]

        for _ in range(10):
            HDML_final, bits_info, snr_measure = HardDecisionMLP(N, snr_dB, device)

            BER_HDML, error_num_HDML = calculate_ber(HDML_final, bits_info)
            if error_num_HDML < 100:
                N += 1000000
                print(f"the code number is {N}")

            else:
                print(
                    f"HD-ML: When SNR is {snr_measure} and signal number is {N}, error number is {error_num_HDML} and BER is {BER_HDML}")
                result[2, i] = BER_HDML
                break


    # Belief Propagation
    for i in range(len(SNR_opt_BP)):
        snr_dB = SNR_opt_BP[i]

        for _ in range(10):
            LDPC_final, bits_info, snr_measure = BeliefPropagation(N, snr_dB, device)

            BER_LDPC, error_num_LDPC = calculate_ber(LDPC_final, bits_info) # BER calculation

            if error_num_LDPC < 100:
                N += 1000000
                print(f"the code number is {N}")

            else:
                print(f"LDPC: When SNR is {snr_measure} and signal number is {N}, error number is {error_num_LDPC} and BER is {BER_LDPC}")
                result[3, i] = BER_LDPC
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

def BeliefPropagation(nr_codeword, snr_dB, device):
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



if __name__ == "__main__":
    device = (torch.device("mps") if torch.backends.mps.is_available()
                                    else (torch.device("cuda") if torch.backends.cuda.is_available()
                                          else torch.device("cpu")))
    # device = torch.device("cpu")

    #Hpyer parameters
    num = int(2e7)
    iter = 5

    result_all = main()
    print(result_all)

    directory_path = "Result"
    csv_filename = "result.csv"
    full_csv_path = os.path.join(directory_path, csv_filename)
    np.savetxt(full_csv_path, result_all, delimiter=',')

    # Data Storage Reference:
    ref_HDML = [0.04368, 0.0331875, 0.02469, 0.01624, 0.01130, 0.00734, 0.00444, 0.00235, 0.00137, 0.000725, 0.000290,
                0.000141944, 4.81731e-05, 1.84191e-05, 5.37554e-06, 1.35552e-06, 3.68058e-07, 6.24751e-08, 9.625e-09,
                1.225e-09, 1.25e-10]

    ref_SDML = [0.0209, 0.014785, 0.00904, 0.0058325, 0.003405, 0.001765, 0.00085975, 0.000385625, 0.000161875,
                6.10714e-05,2.0875e-05, 5.61659e-06, 1.71214e-06, 3.8256e-07, 6.3039e-08, 9.675e-09, 8.25e-10, 0,
                0, 0, 0]
    ref_BP = [0.026185, 0.01830, 0.01246, 0.0076925, 0.0048225, 0.0026775, 0.0013025, 0.0006325, 0.00029950,0.000122045,
              4.875e-05, 1.45349e-05, 4.28082e-06, 1.15101e-06, 2.38981e-07, 3.95057e-08, 6.35e-09, 7e-10, 5e-11, 0, 0]  # ref BP

    uncoded_BPSK = [0.07864588, 0.0670532, 0.05630509, 0.046395885, 0.03750538, 0.02963852, 0.02287882,
                    0.01716848, 0.012505335, 0.00879166, 0.00595442, 0.003866035, 0.00238995, 0.00140195,
                    0.00076981, 0.000398745, 0.000193015, 8.405e-05, 3.335e-05, 1.21e-05, 3.855e-06]  # Uncoded BPSK
    SDML = [0.021438075, 0.014477825, 0.009354125, 0.0056875, 0.003245525, 0.001715425, 0.000867975, 0.000398725,
            0.00016595, 5.9475e-05, 2.1025e-05, 6.275e-06, 1.3875e-06, 3.675e-07, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]
    HDML = [0.04415109, 0.033231805, 0.0241618475, 0.0169061, 0.0113552825, 0.0072749425, 0.00440249,
            0.0025212025, 0.0013590475, 0.0006786375, 0.0003117775, 0.0001306225, 5.03725e-05, 1.759e-05,
            5.2725e-06, 1.4225e-06, 3.425e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]
    BP = [0.0265, 0.018725, 0.013625, 0.0079, 0.0048, 0.003075, 0.0015113636363636364,
          7.136363636363637e-04, 2.659090909090909e-04, 1.2016129032258065e-04, 6.470588235294117e-05, 1.755e-05,
          6.3e-06,
          1.685e-06, 0, 0, 0, 0, 0, 0, 0]

    SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]

    plt.figure(figsize=(10, 10))
    plt.semilogy(SNR, ref_HDML, marker='*', label='ref_HDML')
    plt.semilogy(SNR, ref_SDML, marker='x', label='ref_SDML')
    plt.semilogy(SNR, ref_BP, marker='+', label='ref_BP')
    plt.semilogy(SNR, uncoded_BPSK, marker='.', label='uncoded_BPSK')
    plt.semilogy(SNR, HDML, marker='*', label='HDML')
    plt.semilogy(SNR, SDML, marker='x', label='SDML')
    plt.semilogy(SNR, BP, marker='+', label='BP')

    plt.xlabel('SNR')
    plt.ylabel('BER')
    plt.title('BER Estimation')
    plt.legend(['ref_HDML', 'ref_SDML', 'ref_BP', 'uncoded_BPSK', 'HDML', 'SDML', "BP"], loc='lower left')

    # Display the Plot
    plt.show()