import matplotlib.pyplot as plt

# Data Storage BLER:
article_BPSK = [0.3, 0.25, 0.215, 0.17, 0.15, 0.12, 0.09, # 0~3
                0.069, 0.051, 0.036, 0.025, 0.015, 0.01, 0.0057, 0.0029] # 3.5~7.0
article_SDML = [0.19, 0.15, 0.12, 0.09, 0.065, 0.047, 0.029, # 0~3
                0.0205, 0.013, 0.0072, 0.004, 0.002, 0.001, 0.00035, 0.00012] # 3.5~7.0
article_SLNN_7 = [0.19, 0.15, 0.12, 0.09, 0.065, 0.047, 0.029, # 0~3
                0.0205, 0.013, 0.0072, 0.004, 0.002, 0.001, 0.00035, 0.00012] # 3.5~7.0

BLER_BPSK = [2.779e-01, 2.4299e-01, 2.0673e-01, 1.7226e-01, 1.3924e-01, 1.1376e-01, 8.944e-02, # 0~3
             6.592e-02, 4.948e-02, 3.442e-02, 2.38e-02, 1.482e-02, 9.29e-03, 5.91e-03, 2.94e-03, # 3.5~7
             1.66e-03, 7.238095e-04, # 7.5~8.0
             # 3.15714e-04, 1.3905e-04, 5.23809524e-05, 1.68852459e-05
             ]

BLER_SDML = [0.179194, 0.144548, 0.114578, 0.086722, 0.064089, 0.04507, 0.030437, 0.019683, 0.011838, 0.006678, 0.003697, # 0~5
             0.001767, 0.000793, 0.000334, 0.000119, 4.2e-05, 1.32e-05] # 5.5~8.0

BLER_SLNN_7 = [1.812911e-01, 1.471503e-01, 1.1611e-01, 8.86535e-02, 6.51437e-02, 4.64246e-02, 3.15818e-02, 2.04226e-02, # 0~3.5
               1.25268e-02, 7.151e-03, 3.8876e-03, 1.9449e-03, 9.175e-04, 3.906e-04, 1.467e-04, 5.51e-05, 1.61e-05] # 4.0~8.0


SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8.0]

plt.figure(figsize=(10, 10))
# plt.semilogy(SNR, article_SDML, marker='x', label='Soft-decision ML, Article')
# plt.semilogy(SNR, article_BPSK, marker='+', label='BPSK, Uncoded, Article')
plt.semilogy(SNR, BLER_SDML,  label='Soft-decision ML', color = "black")
plt.semilogy(SNR, BLER_BPSK,  label='BPSK, Uncoded', color = "green")
plt.semilogy(SNR, BLER_SLNN_7, marker='.', label='Single-label Neural network N=7', color='green')

plt.xlabel('SNR')
plt.ylabel('BLER')
plt.title('BLER Estimation')
# plt.legend(['SDML', 'BPSK'], loc='lower left')
# plt.legend(['Soft-decision ML, Article', 'BPSK, Uncoded, Article', 'Soft-decision ML', 'BPSK, Uncoded','Single-label Neural network'], loc='lower left')
plt.legend(['Soft-decision ML', 'BPSK, Uncoded','Single-label Neural network N=7'], loc='lower left')


plt.show()