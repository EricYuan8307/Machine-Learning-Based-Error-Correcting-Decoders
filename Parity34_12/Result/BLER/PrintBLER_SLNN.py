import matplotlib.pyplot as plt

# Data Storage BLER:
article_BPSK = [0.31, 0.29, 0.22, 0.21, 0.17, 0.13, 0.1, 0.08, # 0~3.5
                  0.06, 0.04, 0.029, 0.02, 0.013, 0.0068, 0.004, 0.002, 0.001] # 4.0~8. (10,5) Parity Check code

article_SDML = [0.22, 0.17, 0.102, 0.09, 0.07, 0.04, 0.011, 0.006, # 0~3.5
                  0.0025, 0.0007, 0.0002, 6.5e-05, 1e-05, 2e-06, 4.5e-07, 6e-08, 7e-09] # 4.0~8.0
# article_SLNN_5 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0
# article_SLNN_6 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0
# article_SLNN_7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0

BLER_BPSK_10_5 = [0.3381, 0.2956, 0.2518, 0.211, 0.1841, 0.1377, 0.1135, 0.0818, # 0.0~3.5
             0.0621, 0.0447, 0.0296, 0.0194, 0.0118, 0.007, 0.00384527, 0.00191, 0.0009473] # 4.0~8.0

# BLER_BPSK = [0.6274, 0.56447, 0.49958, 0.43616, 0.36676, 0.30342, 0.242, 0.18795, # 0~3.5
#                0.13963, 0.10029, 0.06936, 0.04605, 0.02726, 0.01616, 0.00882, 0.00445, 0.00222] # 4.0~8.0
BLER_SDML = [0.2167, 0.151, 0.0972, 0.0608, 0.0325, 0.0172, 0.00675, 0.00487, # 0~3.5
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0

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

plt.figure(figsize=(20, 10))
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
plt.title('Parity(34,12) BLER Estimation', fontsize=20)
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