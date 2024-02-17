import matplotlib.pyplot as plt

# Data Storage Reference:
SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8.0]

BER_uncoded_BPSK = [0.078508, 0.067265, 0.056491, 0.046608, 0.037289, 0.029629, 0.022689, 0.017221, # 0~3.5
                0.012463, 0.008816, 0.006007, 0.004012, 0.002327, 0.001388, 0.000743, 0.000386, 0.0002] # 4~8.0

BER_SDML_ref = [0.1, 0.08, 0.055, 0.038, 0.025, 0.015, 0.0078, 0.004, # 0~3.5
                0.0021, 0.00082, 0.0003, 0.0001, 2.9e-5, 7.0e-6, 1.8e-6, 3.5e-7, 5.3e-8] # 4~8.0

BER_SDML = [0.08018, 0.06289, 0.04121, 0.02806, 0.01643, 0.0099, 0.00523, 0.00218, # 0~3.5
                0.0013, 0.0005709, 0.0002209, 7.143e-05, 0.0, 0.0, 0.0, 0.0, 0.0] # 4~8.0

BER_MLNN_100 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0

BER_MLNN_50_50 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0

BER_MLNN_100_100 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0


plt.figure(figsize=(20, 10))
plt.semilogy(SNR, BER_uncoded_BPSK, label='BPSK, Uncoded', color = "green")
plt.semilogy(SNR, BER_SDML, label='Soft-decision ML', color = "black")
plt.semilogy(SNR, BER_SDML_ref, label='Soft-decision ML, Article', color = "black")

# plt.semilogy(SNR, BER_MLNN_100, marker='D', label='N=100', color = "pink", linestyle='--')
# plt.semilogy(SNR, BER_MLNN_50_50, marker='o', label='N1=50, N2=50', color = "orange", linestyle='--')
# plt.semilogy(SNR, BER_MLNN_100_100, marker='v', label='N1=100, N2=100', color = "red", linestyle='--')

plt.xlabel('SNR', fontsize=20)
plt.ylabel('BER', fontsize=20)
plt.title('Parity(26,10) Multi-label Neural Network BER Estimation', fontsize=20)
plt.legend(['BPSK, Uncoded',
            'Soft-decision ML',
            'Soft-decision ML, Article',
            # 'N=100, Article', 'N=100', 'N1=50, N2=50', 'N1=100, N2=100',
            ], loc='lower left')
# plt.legend(['BPSK, Uncoded', 'Soft-decision ML', 'N=100, Article', 'N=100', 'N1=50, N2=50', 'N1=100, N2=100'], loc='lower left')

plt.show()