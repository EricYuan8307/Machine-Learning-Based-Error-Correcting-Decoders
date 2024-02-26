import matplotlib.pyplot as plt

# Data Storage Reference:
SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8.0]

# BER_uncoded_BPSK_ref = [0.078, 0.067, 0.055, 0.0467, 0.036, 0.029, 0.024, 0.017, # 0~3.5
#                 0.013, 0.0088, 0.0059, 0.004, 0.0023, 0.0014, 0.00081, 0.00039, 0.00019] # 4~8.0

BER_uncoded_BPSK_ref = [0.08, 0.067, 0.056, 0.046, 0.038, 0.03, 0.022, 0.017, # 0~3.5
               0.012, 0.0088, 0.0056, 0.00386, 0.0024, 0.0014, 0.00078, 0.00039, 0.0002] # 4.0~8.0

BER_SDML_ref = [0.1, 0.07, 0.045, 0.03, 0.017, 0.009, 0.005, 0.0025, # 0~3.5
                0.001, 0.00045, 9e-05, 2e-05, 3e-06, 6e-07, 1.1e-07, 2e-08, 3.5e-09] # 4.0~8.0

BER_uncoded_BPSK = [0.0787492, 0.0671035, 0.0562085, 0.04638492, 0.0375365, 0.029593083, 0.02294542, 0.01717225, # 0~3.5
               0.01250325, 0.0088127, 0.00597517, 0.00385916, 0.002396, 0.001395, 0.000771, 0.00038875, 0.000196] # 4.0~8.0

BER_SDML = [0.08325, 0.061417, 0.03675, 0.0207, 0.0125, 0.008917, 0.00467, 0.0028056, # 0~3.5
               0.000958, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0

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
plt.title('Parity(30,12) Multi-label Neural Network BER Estimation', fontsize=20)
plt.legend(['BPSK, Uncoded, Article',
            'BPSK, Uncoded',
            'Soft-decision ML, article',
            'Soft-decision ML',
            # 'N=100, Article', 'N=100', 'N1=50, N2=50', 'N1=100, N2=100',
            ], loc='lower left')
# plt.legend(['BPSK, Uncoded', 'Soft-decision ML', 'N=100, Article', 'N=100', 'N1=50, N2=50', 'N1=100, N2=100'], loc='lower left')

plt.show()