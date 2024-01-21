import matplotlib.pyplot as plt

# Data Storage BLER:
BLER_SDML = [0.021445075, 0.01451005, 0.00932015, 0.00566935, 0.0032632, 0.001718075, 0.000866825, 0.000389475,
        0.00016725, 6.15325e-05, 2.06575e-05, 6.2675e-06, 1.6325e-06, 2.825e-07, 0, 0,
        0, 0, 0, 0, 0]

BLER_SLNN = [0.028175, 0.0207, 0.014275, 0.00975, 0.006325, 0.003925, 0.002408125, 0.001395525, 0.00081515, 0.000503175,
        0.000274025, 0.00024585, 0.000102375, 0.000125175, 5.04e-05, 3.4875e-05, 7.05e-06, 1.49e-05, 1.3775e-05, 0, 0]

SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]

plt.figure(figsize=(10, 10))
plt.semilogy(SNR, BLER_SDML, marker='x', label='SDML')
# plt.semilogy(SNR, BLER_SLNN, marker='.', label='SLNN')

plt.xlabel('SNR')
plt.ylabel('BLER')
plt.title('BLER Estimation')
plt.legend(['SDML', 'SLNN'], loc='lower left')

plt.show()