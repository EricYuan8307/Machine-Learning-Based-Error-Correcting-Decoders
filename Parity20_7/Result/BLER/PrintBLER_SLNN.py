import matplotlib.pyplot as plt

# Data Storage BLER:
article_BPSK = [0.31, 0.29, 0.22, 0.21, 0.17, 0.13, 0.1, 0.08, # 0~3.5
                  0.06, 0.04, 0.029, 0.02, 0.013, 0.0068, 0.004, 0.002, 0.001] # 4.0~8. (10,5) Parity Check code
article_SDML = [0.29, 0.25, 0.18, 0.15, 0.11, 0.08, 0.047, 0.03, # 0~3.5
               0.017, 0.0092, 0.0045, 0.002, 8.3e-04, 2.7e-04, 1.01e-04, 3.5e-05, 8.5e-06] # 4.0~8.0

# BLER_BPSK = [0.4311, 0.379, 0.332, 0.2842, 0.2388, 0.1902, 0.1556, 0.1175, # 0~3.5
#              0.0821, 0.0598, 0.0408, 0.0296, 0.0167, 0.0114, 0.005467164, 0.002692, 0.0013985] # 4~8.0
BLER_SDML = [0.2501, 0.1909, 0.1562, 0.1138, 0.0796, 0.0545, 0.0373, 0.0226, # 0~3.5
             0.0123178, 0.006699, 0.003472277, 0.001644554, 0.00074653, 0.00030495, 0.00012475, 0.0, 0.0] # 4~8.0

# BLER_SDML_modified = [0.30568, 0.24994, 0.19757, 0.15134, 0.11428, 0.08054, 0.05285, 0.03511, # 0~3.5
#                0.0218, 0.01194, 0.00713, 0.00335, 0.00162, 0.00080273, 0.000356, 0.0001327, 0.0] # 4.0~8.0 # SNR-0.5

# BLER_SLNN_5 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0
#
# BLER_SLNN_6 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0
#
# BLER_SLNN_7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0
#
# BLER_SLNN_8 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0
#
# BLER_SLNN_9 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0

SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8.0]

plt.figure(figsize=(20, 10))
plt.semilogy(SNR, article_SDML, label='Soft-decision ML, Article', color = "black")
plt.semilogy(SNR, article_BPSK, label='BPSK, Uncoded, Article', color = "green")
plt.semilogy(SNR, BLER_SDML, marker='+', label='Soft-decision ML', color = "black")
# plt.semilogy(SNR, BLER_BPSK, marker='x', label='BPSK, Uncoded', color = "green")

# plt.semilogy(SNR, BLER_SDML_modified, marker='*', label='Soft-decision ML, Modified', color = "black")

# plt.semilogy(SNR, BLER_SLNN_5, marker='x', label='Single-label Neural network N=5', color='blue', linestyle='--')
# plt.semilogy(SNR, BLER_SLNN_6, marker='D', label='Single-label Neural network N=6', color='orange', linestyle='--')
# plt.semilogy(SNR, BLER_SLNN_7, marker='o', label='Single-label Neural network N=7', color='green', linestyle='--')
# plt.semilogy(SNR, BLER_SLNN_8, marker='v', label='Single-label Neural network N=8', color='red', linestyle='--')
# plt.semilogy(SNR, BLER_SLNN_9, marker='<', label='Single-label Neural network N=9', color='purple', linestyle='--')

plt.xlabel('SNR', fontsize=20)
plt.ylabel('BLER', fontsize=20)
plt.title('Parity(20,7) BLER Estimation', fontsize=20)
plt.legend([
    'Soft-decision ML, Article',
    'BPSK, Uncoded, Article',
    'Soft-decision ML',
    # 'BPSK, Uncoded',
            # 'Single-label Neural network N=5', 'Single-label Neural network N=6', 'Single-label Neural network N=7', 'Single-label Neural network N=8', 'Single-label Neural network N=9',
            ], loc='lower left')


plt.show()