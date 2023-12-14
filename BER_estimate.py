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
    result = np.zeros((3,10))
    SNR_opt = [0,1,2,3,4,5,6,7,8,9]
    # SNR_opt = [10, 15]
    N = num
    for i in range(len(SNR_opt)):
        snr_dB =SNR_opt[i]

        # De-Encoder, BPSK only
        for k in range(10):
            modulator = bpsk_modulator()

            bits_info = generator(N)
            modulated_signal = modulator(bits_info.to(mps_device))
            modulated_noise_signal = AWGN(modulated_signal, snr_dB)
            BPSK_final = hard_decision(modulated_noise_signal)

            BER_BPSK, error_num_BPSK= calculate_ber(BPSK_final, bits_info.to(mps_device))

            if error_num_BPSK < 1000:
                N += 10000000
                print(f"the code number is {N}")

            else:
                print(f"BPSK: When SNR is {snr_dB} and signal number is {N}, error number is {error_num_BPSK} and BER is {BER_BPSK}")
                result[0, i] = BER_BPSK
                break

        # Maximum Likelihood
        for m in range(10):
            encoder = hamming_encoder()
            modulator = bpsk_modulator()
            decoder = HammingDecoder()

            # ML:
            bits_info = generator(N)
            encoded_codeword = encoder(bits_info)
            # print("encoded_codeword",encoded_codeword)

            modulated_signal = modulator(encoded_codeword.to(mps_device))
            # print("modulated_signal",modulated_signal)

            modulated_noise_signal = AWGN(modulated_signal, snr_dB)
            # print("modulated_noise_signal",modulated_noise_signal)

            llr_output = llr(modulated_noise_signal, snr_dB)  # Log-Likelihood Calculation
            # print("llr_output",llr_output)

            HD_final = hard_decision(llr_output)
            # print("HD_final",HD_final)
            ML_final = decoder(HD_final)
            # print("ML_final",ML_final)

            # print(ML_final)

            # # Noise Measurment
            # ch_noise = modulated_noise_signal - modulated_signal
            # # Calculate noise power
            # noise_power = torch.mean(ch_noise ** 2)
            #
            # # Calculate practical SNR
            # practical_snr = 10 * torch.log10(1 / (noise_power * 2.0))
            # print('Practical snr: %.2f' % practical_snr.item())

            BER_ML, error_num_ML = calculate_ber(ML_final, bits_info.to(mps_device))

            # print(
            #     f"ML: When SNR is {snr_dB} and total bit number is {N*4}, error number is {error_num_ML} and BER is {BER_ML}")
            # result[1, i] = BER_ML

            if error_num_ML < 1000:
                N += 10000000
                print(f"the code number is {N}")

            else:
                print(
                    f"ML: When SNR is {snr_dB} and signal number is {N}, error number is {error_num_ML} and BER is {BER_ML}")
                result[1, i] = BER_ML
                break

        # BP
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

            if error_num_LDPC < 1000:
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
    num = 10000000 #how many original need to generate
    snr_dB = 10  # Signal-to-noise ratio in dB
    iter = 20 # LDPC Belief Propagation iteration time
    result_all = np.zeros((3, 10))

    result_all = main()
    print(result_all)

    # Create the Plot
    plt.semilogy(result_all.T,marker='x')

    plt.xlabel('SNR')
    plt.ylabel('BER')
    plt.title('Estimation')
    plt.legend(['BPSK', 'ML', 'BP'])

    # Display the Plot
    plt.show()

