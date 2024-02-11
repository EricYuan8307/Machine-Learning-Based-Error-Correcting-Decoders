import matplotlib.pyplot as plt

# Data Storage Reference:
SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8.0]


BER_uncoded_BPSK = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0

BER_SDML = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0

BER_MLNN_100 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0

BER_MLNN_50_50 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0

BER_MLNN_100_100 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0


plt.figure(figsize=(20, 10))
plt.semilogy(SNR, BER_uncoded_BPSK, label='BPSK, Uncoded', color = "green")
plt.semilogy(SNR, BER_SDML, label='Soft-decision ML', color = "black")

# plt.semilogy(SNR, BER_MLNN_100, marker='D', label='N=100', color = "pink", linestyle='--')
# plt.semilogy(SNR, BER_MLNN_50_50, marker='o', label='N1=50, N2=50', color = "orange", linestyle='--')
# plt.semilogy(SNR, BER_MLNN_100_100, marker='v', label='N1=100, N2=100', color = "red", linestyle='--')

plt.xlabel('SNR', fontsize=20)
plt.ylabel('BER', fontsize=20)
plt.title('Parity(34,12) Multi-label Neural Network BER Estimation', fontsize=20)
plt.legend(['BPSK, Uncoded',
            'Soft-decision ML',
            # 'N=100, Article', 'N=100', 'N1=50, N2=50', 'N1=100, N2=100',
            ], loc='lower left')
# plt.legend(['BPSK, Uncoded', 'Soft-decision ML', 'N=100, Article', 'N=100', 'N1=50, N2=50', 'N1=100, N2=100'], loc='lower left')

plt.show()