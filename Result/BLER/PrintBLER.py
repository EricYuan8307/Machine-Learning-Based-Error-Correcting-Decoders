import matplotlib.pyplot as plt

# Data Storage BLER:
BLER_SDML = [0.047267, 0.032096, 0.020736, 0.012686, 0.007356, 0.004019, 0.001937, 0.000878, 0.000396, 0.000133, 5.2e-05, # 0~5
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

BLER_BPSK = [2.779e-01, 2.4299e-01, 2.0673e-01, 1.7226e-01, 1.3924e-01, 1.1376e-01, 8.944e-02, 6.592e-02, 4.948e-02,
             3.442e-02, 2.38e-02, 1.482e-02, 9.29e-03, 5.91e-03, 2.94e-03, 1.66e-03, 7.238095e-04, 3.15714e-04,
             1.3905e-04, 5.23809524e-05, 1.68852459e-05]

BLER_SLNN = [0.048428, 0.033027, 0.021428, 0.013217, 0.007818, 0.00422, 0.002084, 0, 0, 0, 0, # 0~5
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# ref_BLER:
# SNR=1: 1.1e-1,
# SNR=3: 3e-2,
# SNR=5: 3.5e-3,
# SNR=7: 1.4e-4.

BLER_SLNN_OF = [0.06341, 0.04639, 0.03226, 0.02167, 0.01385, 0.00832, 0.00505, 0.00276, 0.00123, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]

plt.figure(figsize=(10, 10))
plt.semilogy(SNR, BLER_SDML, marker='x', label='SDML')
plt.semilogy(SNR, BLER_BPSK, marker='+', label='BPSK')
plt.semilogy(SNR, BLER_SLNN, marker='.', label='SLNN')
plt.semilogy(SNR, BLER_SLNN_OF, marker='.', label='overfitting SLNN')

plt.xlabel('SNR')
plt.ylabel('BLER')
plt.title('BLER Estimation')
# plt.legend(['SDML', 'BPSK'], loc='lower left')
plt.legend(['SDML', 'BPSK','SLNN', 'Overfitting SLNN'], loc='lower left')

plt.show()