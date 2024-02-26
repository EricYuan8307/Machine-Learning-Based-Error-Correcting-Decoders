import matplotlib.pyplot as plt

# Data Storage Reference:
SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8.0]

BER_uncoded_BPSK_ref = [0.078, 0.066, 0.058, 0.045, 0.039, 0.029, 0.021, 0.017, # 0~3.5
               0.012, 0.009, 0.006, 0.004, 0.0025, 0.0015, 0.0008, 0.00038, 0.00019] # 4.0~8.0

BER_uncoded_BPSK = [0.0788417, 0.0661083, 0.0562583, 0.046975, 0.0377583, 0.03035, 0.023617, 0.017125, # 0~3.5
               0.0126, 0.00877, 0.006025, 0.00387, 0.0025083, 0.001583, 0.000772, 0.000396, 0.0001934] # 4.0~8.0

BER_SDML_ref = [0.1, 0.08, 0.06, 0.04, 0.027, 0.016, 0.009, 0.0049, # 0~3.5
               0.0016, 0.00061, 0.0002, 5e-05, 1.2e-05, 2e-06, 4e-07, 7e-08, 1e-08] # 4.0~8.0

BER_SDML = [0.081625, 0.0563083, 0.037517, 0.024483, 0.0136583, 0.0069583, 0.00313, 0.001517, # 0~3.5
               0.0006375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0

# BER_MLNN_100 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0
#
# BER_MLNN_50_50 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0
#
# BER_MLNN_100_100 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0


plt.figure(figsize=(16, 9))
plt.semilogy(SNR, BER_uncoded_BPSK_ref, label='BPSK, Uncoded, Article', color = "green")
plt.semilogy(SNR, BER_uncoded_BPSK, label='BPSK, Uncoded', color = "green")
plt.semilogy(SNR, BER_SDML_ref, label='Soft-decision ML, Article', color = "black")
plt.semilogy(SNR, BER_SDML, marker='+', label='Soft-decision ML', color = "black")

# plt.semilogy(SNR, BER_MLNN_100, marker='D', label='N=100', color = "pink", linestyle='--')
# plt.semilogy(SNR, BER_MLNN_50_50, marker='o', label='N1=50, N2=50', color = "orange", linestyle='--')
# plt.semilogy(SNR, BER_MLNN_100_100, marker='v', label='N1=100, N2=100', color = "red", linestyle='--')

plt.xlabel('SNR', fontsize=20)
plt.ylabel('BER', fontsize=20)
plt.title('Parity(34,12) Multi-label Neural Network BER Estimation', fontsize=20)
plt.legend([
    'BPSK, Uncoded, Article',
    'BPSK, Uncoded',
    'Soft-decision ML, Article',
    'Soft-decision ML',
            # 'N=100, Article', 'N=100', 'N1=50, N2=50', 'N1=100, N2=100',
            ], loc='lower left')
# plt.legend(['BPSK, Uncoded', 'Soft-decision ML', 'N=100, Article', 'N=100', 'N1=50, N2=50', 'N1=100, N2=100'], loc='lower left')

plt.show()