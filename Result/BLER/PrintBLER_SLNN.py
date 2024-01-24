import matplotlib.pyplot as plt

# Data Storage BLER:
article_BPSK = [0.3, 0.25, 0.215, 0.17, 0.15, 0.12, 0.09, # 0~3
                0.069, 0.051, 0.036, 0.025, 0.015, 0.01, 0.0057, # 3.5~6.5
                0.0029] #
article_SDML = [0, 0, 0, 0, 0, 0, 0, # 0~3
                0, 0, 0, 0, 0, 0, 0, # 3.5~6.5
                0]


BLER_BPSK = [2.779e-01, 2.4299e-01, 2.0673e-01, 1.7226e-01, 1.3924e-01, 1.1376e-01, 8.944e-02, # 0~3
             6.592e-02, 4.948e-02, 3.442e-02, 2.38e-02, 1.482e-02, 9.29e-03, 5.91e-03, 2.94e-03, # 3.5~7
             # 1.66e-03, 7.238095e-04, 3.15714e-04, 1.3905e-04, 5.23809524e-05, 1.68852459e-05
             ]

BLER_SDML = [0.179194, 0.144548, 0.114578, 0.086722, 0.064089, 0.04507, 0.030437, 0.019683, # 0~3
             0.011838, 0.006678, 0.003697, 0.001767, 0.000793, 0.000334, # 3.5~6.5
             0.000119,
             # 4.2e-05, 1.32e-05
             ]

BLER_SLNN = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # 0~5
             0, 0, 0, 0]


SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]

plt.figure(figsize=(10, 10))
plt.semilogy(SNR, article_SDML, marker='x', label='article_SDML')
plt.semilogy(SNR, article_BPSK, marker='+', label='article_BPSK')
plt.semilogy(SNR, BLER_SDML, marker='x', label='SDML')
plt.semilogy(SNR, BLER_BPSK, marker='+', label='BPSK')
plt.semilogy(SNR, BLER_SLNN, marker='.', label='SLNN')

plt.xlabel('SNR')
plt.ylabel('BLER')
plt.title('BLER Estimation')
# plt.legend(['SDML', 'BPSK'], loc='lower left')
plt.legend(['article_SDML', 'article_BPSK','SDML', 'BPSK','SLNN'], loc='lower left')

plt.show()