import matplotlib.pyplot as plt

# Data Storage BLER:
article_BPSK = [0.31, 0.29, 0.22, 0.21, 0.17, 0.13, 0.1, 0.08, # 0~3.5
                  0.06, 0.04, 0.029, 0.02, 0.013, 0.0068, 0.004, 0.002, 0.001] # 4.0~8. (10,5) Parity Check code
article_SDML = [0.29, 0.21, 0.15, 0.103, 0.075, 0.041, 0.022, 0.012, # 0~3.5
                  0.006, 0.0029, 0.0011, 0.00027, 0.0001, 1.9e-05, 5.2e-06, 1.2e-06, 2.5e-07] # 4.0~8.0
# article_SLNN_5 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0
# article_SLNN_6 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0
# article_SLNN_7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0

# BLER_BPSK = [0.33704, 0.29044, 0.25275, 0.21242, 0.17261, 0.13949, 0.10805, 0.08315, # 0~3.5
#                0.05994, 0.04359, 0.02895, 0.01934, 0.01193, 0.00698, 0.00385, 0.00204, 0.00103] # 4.0~8.0

BLER_BPSK = [0.55953, 0.50223, 0.4406, 0.37664, 0.31537, 0.25889, 0.20519, 0.15945, # 0~3.5
               0.11863, 0.08547, 0.05748, 0.039, 0.02289, 0.0138, 0.00719, 0.00373, 0.00208] # 4.0~8.0

BLER_SDML = [0.2778, 0.2169, 0.16, 0.114, 0.07, 0.0425, 0.0229, 0.01295, # 0~3.5
               0.00515, 0.00288, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0

# BLER_SDML = [0.274, 0.204, 0.145, 0.09736, 0.06127, 0.0375, 0.0216, 0.0107, # 0~3.5
#                0.0049, 0.0029, 0.001175, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0 # Result from old Generator matrix

BLER_BPSK_10_5 = [0.3381, 0.2956, 0.2518, 0.211, 0.1841, 0.1377, 0.1135, 0.0818, # 0.0~3.5
             0.0621, 0.0447, 0.0296, 0.0194, 0.0118, 0.007, 0.00384527, 0.00191, 0.0009473] # 4.0~8.0

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
plt.semilogy(SNR, article_SDML, label='Soft-decision ML, Article', color = "black")
plt.semilogy(SNR, article_BPSK, label='BPSK, Uncoded, Article', color = "green")
plt.semilogy(SNR, BLER_SDML, marker='+', label='Soft-decision ML', color = "black")
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
plt.title('Parity(26,10) BLER Estimation', fontsize=20)
plt.legend(['Soft-decision ML, Article',
            'BPSK, Uncoded, Article',
            # 'Soft-decision ML',
            'BPSK, Uncoded',
            'Parity(10,5) BPSK, Uncoded',
            # 'Single-label Neural network N=5, Article', 'Single-label Neural network N=6, Article', 'Single-label Neural network N=7, Article',
            # 'Single-label Neural network N=5', 'Single-label Neural network N=6', 'Single-label Neural network N=7', 'Single-label Neural network N=8', 'Single-label Neural network N=9',
            ], loc='lower left')


plt.show()