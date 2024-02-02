import matplotlib.pyplot as plt

# Data Storage BLER:
# article_MLNN_100 = [0.34, 0.31, 0.26, 0.23, 0.19, 0.16, 0.14, 0.105, # 0~3.5
#                     0.079, 0.062, 0.04, 0.03, 0.021, 0.014, 0.0085, # 4.0~7.0
#                     5e-03, 3.1e-03] # 7.5~8.0
# article_MLNN_50_50 = [0.25, 0.21, 0.18, 0.15, 0.12, 0.089, 0.061, 0.048, # 0~3.5
#                       0.033, 0.021, 0.014, 0.0087, 0.0047, 0.0023, 0.0014, # 4.0~7.0
#                       6e-04, 2.7e-04] # 7.5~8.0
# article_MLNN_100_100 = [0.19, 0.15, 0.12, 0.09, 0.065, 0.047, 0.029, # 0~3
#                         0.0205, 0.013, 0.0072, 0.004, 0.002, 0.001, 0.00035, 0.00014, # 3.5~7.0
#                         5.31e-05, 1.6e-05] # 7.5~8.0

BLER_BPSK = [2.779e-01, 2.4299e-01, 2.0673e-01, 1.7226e-01, 1.3924e-01, 1.1376e-01, 8.944e-02, # 0~3
             6.592e-02, 4.948e-02, 3.442e-02, 2.38e-02, 1.482e-02, 9.29e-03, 5.91e-03, 2.94e-03, # 3.5~7
             1.66e-03, 7.238095e-04] # 7.5~8.0
BLER_SDML = [0.179194, 0.144548, 0.114578, 0.086722, 0.064089, 0.04507, 0.030437, 0.019683, 0.011838, 0.006678, 0.003697, # 0~5
             0.001767, 0.000793, 0.000334, 0.000119, 4.2e-05, 1.32e-05] # 5.5~8.0

BLER_SLNN_7 = [1.812911e-01, 1.471503e-01, 1.1611e-01, 8.86535e-02, 6.51437e-02, 4.64246e-02, 3.15818e-02, 2.04226e-02, # 0~3.5
               1.25268e-02, 7.151e-03, 3.8876e-03, 1.9449e-03, 9.175e-04, 3.906e-04, 1.467e-04, 5.51e-05, 1.61e-05] # 4.0~8.0
BLER_MLNN_100 = [0.210868, 0.1746373, 0.1417056, 0.1121143, 0.0862392, 0.0643132, 0.0465085, 0.0323077, # 0~3.5
               0.0215793, 0.0138018, 0.0083611, 0.0048484, 0.0026326, 0.0013477, 0.0006301, 0.0002772, 0.000109] # 4.0~8.0
BLER_MLNN_50_50 = [0.1990236, 0.1632408, 0.1305324, 0.1013625, 0.0762221, 0.0552524, 0.0383573, 0.0255042, # 0~3.5
                   0.0160266, 0.0094852, 0.0053169, 0.0027371, 0.001297, 0.0005622, 0.00023293, 8.11e-05, 2.54e-05] # 4.0~8.0
BLER_MLNN_100_100 = [0.1965366, 0.1609105, 0.1286232, 0.0996521, 0.0744929, 0.0537661, 0.0370903, 0.024426, # 0~3.5
                     0.015206, 0.0089872, 0.0049126, 0.0024766, 0.0011737, 0.0005027, 0.0001987, 6.86e-05, 2.05e-05] # 4.0~8.0

SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8.0]

plt.figure(figsize=(10, 10))
plt.semilogy(SNR, BLER_SDML,  label='Soft-decision ML', color = "black")
plt.semilogy(SNR, BLER_BPSK,  label='BPSK, Uncoded', color = "green")

# plt.semilogy(SNR, article_MLNN_100, marker='D', label='N=100, Article', color = "pink", linestyle='--')
# plt.semilogy(SNR, article_MLNN_50_50, marker='D', label='N1=50, N2=50, Article', color = "orange", linestyle='--')
# plt.semilogy(SNR, article_MLNN_100_100, marker='D', label='N1=100, N2=100, Article', color = "red", linestyle='--')

plt.semilogy(SNR, BLER_SLNN_7, label='Single-label Neural network N=7', color='blue')
plt.semilogy(SNR, BLER_MLNN_100, marker='D', label='N=100', color = "pink", linestyle='--')
plt.semilogy(SNR, BLER_MLNN_50_50, marker='D', label='N1=50, N2=50', color = "orange", linestyle='--')
plt.semilogy(SNR, BLER_MLNN_100_100, marker='D', label='N1=100, N2=100', color = "red", linestyle='--')

plt.xlabel('SNR')
plt.ylabel('BLER')
plt.title('BLER Estimation')
plt.legend(['Soft-decision ML', 'BPSK, Uncoded',
            # 'N=100, Article', 'N1=50, N2=50, Article', 'N1=100, N2=100, Article',
            'Single-label Neural network N=7', 'N=100', 'N1=50, N2=50', 'N1=100, N2=100'], loc='lower left')


plt.show()