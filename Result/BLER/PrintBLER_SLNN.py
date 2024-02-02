import matplotlib.pyplot as plt

# Data Storage BLER:
article_BPSK = [0.3, 0.25, 0.215, 0.17, 0.15, 0.12, 0.09, # 0~3
                0.069, 0.051, 0.036, 0.025, 0.015, 0.01, 0.0057, 0.0029, # 3.5~7.0
                0.0017, 0.00073] # 7.5~8.0
article_SDML = [0.19, 0.15, 0.12, 0.09, 0.065, 0.047, 0.029, # 0~3
                0.0205, 0.013, 0.0072, 0.004, 0.002, 0.001, 0.00035, 0.00012, # 3.5~7.0
                0.000042, 0.000013] # 7.5~8.0
article_SLNN_5 = [0.34, 0.31, 2.71075e-01, 2.343207e-01, 1.98604e-01, 1.64944e-01, 1.341949e-01, 1.06558e-01, # 0~3.5
                  8.24865e-02, 6.20837e-02, 4.52981e-02, 3.21064e-02, 2.20062e-02, 1.45313e-02, 9.2026e-03, # 4.0~7.0
                  5.6337e-03, 3.3177e-03] # 7.5~8.0
article_SLNN_6 = [2.53975e-01, 2.163056e-01, 1.799663e-01, 1.467614e-01, 1.164633e-01, 9.00929e-02, 6.72701e-02, 4.86077e-02, # 0~3.5
                  3.37757e-02, 2.25114e-02, 1.43565e-02, 8.6979e-03, 5.0099e-03, 2.7493e-03, 1.3948e-03, # 4.0~7.0
                  6.763e-04, 3.041e-04] # 7.5~8.0
article_SLNN_7 = [0.19, 0.15, 0.12, 0.09, 0.065, 0.047, 0.029, # 0~3
                  0.0205, 0.013, 0.0072, 0.004, 0.002, 0.001, 0.00035, 0.00014, # 3.5~7.0
                  5.31e-05, 1.6e-05] # 7.5~8.0



BLER_BPSK = [2.779e-01, 2.4299e-01, 2.0673e-01, 1.7226e-01, 1.3924e-01, 1.1376e-01, 8.944e-02, # 0~3
             6.592e-02, 4.948e-02, 3.442e-02, 2.38e-02, 1.482e-02, 9.29e-03, 5.91e-03, 2.94e-03, # 3.5~7
             1.66e-03, 7.238095e-04] # 7.5~8.0
BLER_SDML = [0.179194, 0.144548, 0.114578, 0.086722, 0.064089, 0.04507, 0.030437, 0.019683, 0.011838, 0.006678, 0.003697, # 0~5
             0.001767, 0.000793, 0.000334, 0.000119, 4.2e-05, 1.32e-05] # 5.5~8.0

BLER_SLNN_5 = [3.477188e-01, 3.093166e-01, 2.71075e-01, 2.343207e-01, 1.98604e-01, 1.64944e-01, 1.341949e-01, 1.06558e-01, # 0~3.5
               8.24865e-02, 6.20837e-02, 4.52981e-02, 3.21064e-02, 2.20062e-02, 1.45313e-02, 9.2026e-03, 5.6337e-03, 3.3177e-03] # 4.0~8.0
BLER_SLNN_6 = [2.53975e-01, 2.163056e-01, 1.799663e-01, 1.467614e-01, 1.164633e-01, 9.00929e-02, 6.72701e-02, 4.86077e-02, # 0~3.5
               3.37757e-02, 2.25114e-02, 1.43565e-02, 8.6979e-03, 5.0099e-03, 2.7493e-03, 1.3948e-03, 6.763e-04, 3.041e-04] # 4.0~8.0
BLER_SLNN_7 = [1.812911e-01, 1.471503e-01, 1.1611e-01, 8.86535e-02, 6.51437e-02, 4.64246e-02, 3.15818e-02, 2.04226e-02, # 0~3.5
               1.25268e-02, 7.151e-03, 3.8876e-03, 1.9449e-03, 9.175e-04, 3.906e-04, 1.467e-04, 5.51e-05, 1.61e-05] # 4.0~8.0
BLER_SLNN_8 = [1.826333e-01, 1.482811e-01, 1.170199e-01, 8.96328e-02, 6.60949e-02, 4.71128e-02, 3.21477e-02, 2.08218e-02, # 0~3.5
               1.28694e-02, 7.4719e-03, 4.077e-03, 2.0556e-03, 9.563e-04, 4.026e-04, 1.575e-04, 5.98e-05, 1.78e-05] # 4.0~8.0
BLER_SLNN_9 = [1.818465e-01, 1.477253e-01, 1.165583e-01, 8.91116e-02, 6.56956e-02, 4.68068e-02, 3.17989e-02, 2.05643e-02, # 0~3.5
               1.26734e-02, 7.3259e-03, 3.9692e-03, 2.0027e-03, 9.375e-04, 4.063e-04, 1.508e-04, 5.46e-05, 1.64e-05,] # 4.0~8.0

SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8.0]

plt.figure(figsize=(10, 10))
# plt.semilogy(SNR, article_SDML, marker='x', label='Soft-decision ML, Article')
# plt.semilogy(SNR, article_BPSK, marker='+', label='BPSK, Uncoded, Article')
plt.semilogy(SNR, BLER_SDML,  label='Soft-decision ML', color = "black")
plt.semilogy(SNR, BLER_BPSK,  label='BPSK, Uncoded', color = "green")

plt.semilogy(SNR, article_SLNN_5, marker='x', label='Single-label Neural network N=5, Article', color='blue', linestyle='--')
plt.semilogy(SNR, article_SLNN_6, marker='D', label='Single-label Neural network N=6, Article', color='orange', linestyle='--')
plt.semilogy(SNR, article_SLNN_7, marker='o', label='Single-label Neural network N=7, Article', color='green', linestyle='--')

plt.semilogy(SNR, BLER_SLNN_5, marker='x', label='Single-label Neural network N=5', color='blue', linestyle='--')
plt.semilogy(SNR, BLER_SLNN_6, marker='D', label='Single-label Neural network N=6', color='orange', linestyle='--')
plt.semilogy(SNR, BLER_SLNN_7, marker='o', label='Single-label Neural network N=7', color='green', linestyle='--')
plt.semilogy(SNR, BLER_SLNN_8, marker='v', label='Single-label Neural network N=8', color='red', linestyle='--')
plt.semilogy(SNR, BLER_SLNN_9, marker='<', label='Single-label Neural network N=9', color='purple', linestyle='--')

plt.xlabel('SNR')
plt.ylabel('BLER')
plt.title('BLER Estimation')
# plt.legend(['SDML', 'BPSK'], loc='lower left')
# plt.legend(['Soft-decision ML, Article', 'BPSK, Uncoded, Article', 'Soft-decision ML', 'BPSK, Uncoded','Single-label Neural network'], loc='lower left')
plt.legend(['Soft-decision ML', 'BPSK, Uncoded',
            'Single-label Neural network N=5, Article', 'Single-label Neural network N=6, Article', 'Single-label Neural network N=7, Article',
            'Single-label Neural network N=5', 'Single-label Neural network N=6', 'Single-label Neural network N=7', 'Single-label Neural network N=8', 'Single-label Neural network N=9', ], loc='lower left')


plt.show()