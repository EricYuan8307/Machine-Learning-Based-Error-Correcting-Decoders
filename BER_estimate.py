from unittest import result

import matplotlib.pyplot as plt
import torch
import numpy as np

from Encoder.BPSK import bpsk_modulator
from Encoder.Hamming74 import hamming_encoder
from Decoder.likelihood import llr
from Decoder.LDPC_BP import LDPCBeliefPropagation
from Decoder.HardDecision import hard_decision
from Transmit.noise import AWGN
from Estimation.BitErrorRate import calculate_ber
from Decoder.HammingDecoder import HammingDecoder


# Code Generation
def generator(nr_codewords):
    bits = torch.randint(0, 2, size=(nr_codewords, 1, 4), dtype=torch.int)
    # bits = torch.tensor([[[1, 1, 0, 1]],
    #                      [[0, 1, 0, 0,]]], dtype=torch.int32)

    return bits


# Calculate the Error number and BER
def main():
    SNR_opt_BPSK = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    SNR_opt_ML = [0, 1, 2, 3, 4, 5, 6, 7]
    SNR_opt_BP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    result = np.zeros((3, len(SNR_opt_BPSK)))
    N = num

    # # De-Encoder, BPSK only
    # for i in range(len(SNR_opt_BPSK)):
    #     snr_dB =SNR_opt_BPSK[i]
    #
    #     for j in range(10):
    #         modulator = bpsk_modulator()
    #
    #         bits_info = generator(N)
    #         modulated_signal = modulator(bits_info.to(mps_device))
    #         modulated_noise_signal = AWGN(modulated_signal, snr_dB)
    #         BPSK_final = hard_decision(modulated_noise_signal)
    #
    #         BER_BPSK, error_num_BPSK= calculate_ber(BPSK_final, bits_info.to(mps_device))
    #
    #         if error_num_BPSK < 100:
    #             N += 10000000
    #             print(f"the code number is {N}")
    #
    #         else:
    #             print(f"BPSK: When SNR is {snr_dB} and signal number is {N}, error number is {error_num_BPSK} and BER is {BER_BPSK}")
    #             result[0, i] = BER_BPSK
    #             break

    # # Maximum Likelihood
    # for i in range(len(SNR_opt_ML)):
    #     snr_dB = SNR_opt_ML[i]
    #
    #     for j in range(10):
    #         encoder = hamming_encoder()
    #         modulator = bpsk_modulator()
    #         decoder = HammingDecoder()
    #
    #         # ML:
    #         bits_info = generator(N)
    #         encoded_codeword = encoder(bits_info)
    #
    #         modulated_signal = modulator(encoded_codeword.to(mps_device))
    #
    #         modulated_noise_signal = AWGN(modulated_signal, snr_dB)
    #
    #         llr_output = llr(modulated_noise_signal, snr_dB)  # Log-Likelihood Calculation
    #
    #         HD_final = hard_decision(llr_output)
    #         ML_final = decoder(HD_final)
    #
    #         BER_ML, error_num_ML = calculate_ber(ML_final, bits_info.to(mps_device))
    #
    #         if error_num_ML < 100 & N <= 40000000:
    #             N += 10000000
    #             print(f"the code number is {N}")
    #
    #         else:
    #             print(
    #                 f"ML: When SNR is {snr_dB} and signal number is {N}, error number is {error_num_ML} and BER is {BER_ML}")
    #             result[1, i] = BER_ML
    #             break

    # Belief Propagation
    for i in range(len(SNR_opt_BP)):
        snr_dB = SNR_opt_BP[i]

        for j in range(30):

            encoder = hamming_encoder()
            modulator = bpsk_modulator()
            decoder = HammingDecoder()

            bits_info = generator(N)  # Code Generator
            encoded_codeword = encoder(bits_info) # Hamming(7,4) Encoder
            modulated_signal = modulator(encoded_codeword.to(mps_device)) # Modulate signal
            modulated_noise_signal = AWGN(modulated_signal, snr_dB) # Add Noise
            llr_output = llr(modulated_noise_signal, snr_dB) #LLR

            ldpc_bp = LDPCBeliefPropagation(llr_output.to(mps_device))
            LDPC_result = ldpc_bp(iter) # LDPC
            LDPC_HD = hard_decision(LDPC_result) #Hard Decision
            LDPC_final = decoder(LDPC_HD)

            BER_LDPC, error_num_LDPC = calculate_ber(LDPC_final, bits_info.to(mps_device)) # BER calculation

            if error_num_LDPC < 100:
                N += 10000000
                print(f"the code number is {N}")

            else:
                print(f"LDPC: When SNR is {snr_dB} and signal number is {N}, error number is {error_num_LDPC} and BER is {BER_LDPC}")
                result[2, i] = BER_LDPC
                break



    return result


if __name__ == "__main__":
    mps_device = (torch.device("mps") if torch.backends.mps.is_available()
                                    else (torch.device("cuda") if torch.backends.cuda.is_available()
                                          else torch.device("cpu")))

    #Hpyer parameters
    num = int(1e7) #how many original need to generate
    iter = 5 # LDPC Belief Propagation iteration time

    # Store result for plot
    result_all = np.zeros((3, 10))

    result_all = main()

    # Create the Plot
    plt.semilogy(result_all.T,marker='*')

    plt.xlabel('SNR')
    plt.ylabel('BER')
    plt.title('Estimation')
    plt.legend(['BPSK', 'ML', 'BP'])

    # Display the Plot
    plt.show()

