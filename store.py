#
# # Calculate the Error number and BER
# def main():
#
#     SNR_opt = [0,1,2,3,4,5,10,15]
#     N = 1000000
#
#     for i in range(len(SNR_opt)):
#         snr_dB =SNR_opt[i]
#         for j in range(10):
#             num = 1000000
#
#             # Count Error Number and BER:
#             encoder = hamming_encoder()  # Generate Encoded Data with 3 parity bits
#             modulator = bpsk_modulator() # Modulate the signal
#
#
#             # Full LDPC
#             bits_info = generator(num)
#             encoded_codeword = encoder(bits_info)
#             modulated_noise_signal = modulator(encoded_codeword.to(mps_device), snr_dB)
#             llr_output = llr(modulated_noise_signal, snr_dB)# Log-Likelihood Calculation
#             ldpc_bp = LDPCBeliefPropagation(llr_output.to(mps_device))  # LDPC Belief Propagation
#             LDPC_result = ldpc_bp(iter==20) # BP iteration time
#             LDPC_final = hard_decision_cutter(LDPC_result)
#
#             bits_info = bits_info.to(mps_device)
#             error_num_LDPC, BER_LDPC = calculate_ber(LDPC_final, bits_info)
#
#             # if error_num_LDPC < 1000:
#             #     num += 5000
#
#             if error_num_LDPC >= 1000:
#             # elif error_num_LDPC >= 1000:
#                 print(f"LDPC: When SNR is {snr_dB} and signal number is {num}, error number is {error_num_LDPC} and BER is {BER_LDPC}")
#                 break
#
#
#
#         # for k in range(10):
#         #     num = N + 5000 * k
#         #
#         #     # De-Encoder, BPSK only
#         #     bits_info = generator(num)
#         #     modulated_noise_signal = modulator(bits_info.to(mps_device), snr_dB)
#         #
#         #     bits_info = bits_info.to(mps_device)  # bits_info: original signal
#         #     error_num_BPSK, BER_BPSK = calculate_ber(modulated_noise_signal, bits_info)
#         #     print(f"LDPC: Error number is {error_num_BPSK} and BER is {BER_BPSK}")
#         #
#         #
#         # for m in range(10):
#         #     num = N + 5000 * m
#         #     # Maximum Likelihood
#         #     bits_info = generator(num)
#         #     encoded_codeword = encoder(bits_info)
#         #     modulated_noise_signal = modulator(encoded_codeword.to(mps_device), snr_dB)
#         #     llr_output = llr(modulated_noise_signal, snr_dB)  # Log-Likelihood Calculation
#         #     ML_final = hard_decision_cutter(llr_output)
#         #
#         #     bits_info = bits_info.to(mps_device)  # bits_info: original signal
#         #     error_num_ML, BER_ML = calculate_ber(ML_final, bits_info)
#         #     print(f"LDPC: Error number is {error_num_ML} and BER is {BER_ML}")
#
#
#
#             # if error_num > 1000:
#             #     print("snr_dB:", snr_dB)
#             #     print(error_num)
#             #     print(BER)
#             #
#             #     break
#             # elif error_num <1000:
#             #     continue
#
#
# if __name__ == "__main__":
#     main()