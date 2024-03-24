import matplotlib.pyplot as plt

# Data Storage BLER:
article_BPSK = [0.31, 0.29, 0.22, 0.21, 0.17, 0.13, 0.1, 0.08, # 0~3.5
                  0.06, 0.04, 0.029, 0.02, 0.013, 0.0068, 0.004, 0.002, 0.001] # 4.0~8. (10,5) Parity Check code

article_SDML = [0.25, 0.18, 0.13, 0.09, 0.06, 0.039, 0.02, 0.01, # 0~3.5
                  0.0045, 0.002, 0.00075, 0.00021, 6e-05, 2e-05, 4e-06, 1.02e-06, 1.6e-07] # 4.0~8.0
# article_SLNN_5 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0
# article_SLNN_6 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0
# article_SLNN_7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0

BLER_BPSK_10_5 = [0.3381, 0.2956, 0.2518, 0.211, 0.1841, 0.1377, 0.1135, 0.0818, # 0.0~3.5
             0.0621, 0.0447, 0.0296, 0.0194, 0.0118, 0.007, 0.00384527, 0.00191, 0.0009473] # 4.0~8.0

BLER_BPSK = [0.62717, 0.56514, 0.49935, 0.43237, 0.36873, 0.30276, 0.24216, 0.19092, # 0~3.5
               0.14016, 0.10127, 0.06888, 0.04543, 0.02831, 0.01718, 0.00907, 0.00505, 0.00238] # 4.0~8.0

BLER_SDML = [0.2534, 0.1815, 0.1295, 0.083, 0.0508, 0.0279, 0.0135, 0.0069, # 0~3.5
               0.00275, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0

# BLER_SLNN_5 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0
# BLER_SLNN_6 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0
# BLER_SLNN_7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0
# BLER_SLNN_8 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0
# BLER_SLNN_9 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0

SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8.0]

plt.figure(figsize=(16, 9))
plt.semilogy(SNR, article_SDML,  label='Soft-decision ML, Article', color = "black")
plt.semilogy(SNR, article_BPSK,  label='BPSK, Uncoded, Article', color = "green")
plt.semilogy(SNR, BLER_SDML, marker='x', label='Soft-decision ML', color = "black")
# plt.semilogy(SNR, BLER_BPSK, marker='+', label='BPSK, Uncoded', color = "green")

plt.semilogy(SNR, BLER_BPSK_10_5, label='Parity(10,5) BPSK, Uncoded', color = "green")


# plt.semilogy(SNR, article_SLNN_5, marker='x', label='Single-label Neural network N=5, Article', color='blue', linestyle='--')
# plt.semilogy(SNR, article_SLNN_6, marker='D', label='Single-label Neural network N=6, Article', color='orange', linestyle='--')
# plt.semilogy(SNR, article_SLNN_7, marker='o', label='Single-label Neural network N=7, Article', color='green', linestyle='--')
#
# plt.semilogy(SNR, BLER_SLNN_5, marker='x', label='Single-label Neural network N=5', color='blue', linestyle='--')
# plt.semilogy(SNR, BLER_SLNN_6, marker='D', label='Single-label Neural network N=6', color='orange', linestyle='--')
# plt.semilogy(SNR, BLER_SLNN_7, marker='o', label='Single-label Neural network N=7', color='green', linestyle='--')
# plt.semilogy(SNR, BLER_SLNN_8, marker='v', label='Single-label Neural network N=8', color='red', linestyle='--')
# plt.semilogy(SNR, BLER_SLNN_9, marker='<', label='Single-label Neural network N=9', color='purple', linestyle='--')

plt.xlabel('SNR', fontsize=20)
plt.ylabel('BLER', fontsize=20)
plt.title('Parity(30,12) BLER Estimation', fontsize=20)
plt.legend([
    'Soft-decision ML, Article',
    'BPSK, Uncoded, Article',
    'Soft-decision ML',
    # 'BPSK, Uncoded',
    'Parity(10,5) BPSK, Uncoded',
    # 'Single-label Neural network N=5, Article', 'Single-label Neural network N=6, Article', 'Single-label Neural network N=7, Article',
            # 'Single-label Neural network N=5', 'Single-label Neural network N=6', 'Single-label Neural network N=7', 'Single-label Neural network N=8', 'Single-label Neural network N=9',
            ], loc='lower left')


plt.show()