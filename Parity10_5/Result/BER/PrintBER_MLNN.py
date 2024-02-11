import matplotlib.pyplot as plt

# Data Storage Reference:
SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8.0]

BER_uncoded_BPSK = [0.07834, 0.06766, 0.0563, 0.04808, 0.03832, 0.03056, 0.02334, 0.01754, # 0~3.5
                0.0129, 0.00862, 0.00604, 0.0037, 0.0028, 0.0013996, 0.00077552, 0.000408159, 0.0002008 # 4.0~8.0
                ]
BER_SDML = [0.07526, 0.06006, 0.04646, 0.03424, 0.0252, 0.01648, 0.01282, 0.00766, # 0~3.5
        0.00416, 0.0034, 0.0014, 0.000729, 0.00035, 0.000153, 5.82178e-05, 2.277e-05, 7.44186e-06 # 4.0~8.0
        ]

BER_MLNN_100 = [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, # 0~3.5
        0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0] # 4~8.0

BER_MLNN_50_50 = [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, # 0~3.5
        0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0] # 4~8.0

BER_MLNN_100_100 = [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, # 0~3.5
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0] # 4~8.0


def nnplot():
    plt.figure(figsize=(20, 10))
    plt.semilogy(SNR, BER_uncoded_BPSK, label='BPSK, Uncoded', color = "green")
    plt.semilogy(SNR, BER_SDML, label='Soft-decision ML', color = "black")
    # plt.semilogy(SNR, BER_MLNN_100, marker='D', label='N=100', color = "pink", linestyle='--')
    # plt.semilogy(SNR, BER_MLNN_50_50, marker='o', label='N1=50, N2=50', color = "orange", linestyle='--')
    # plt.semilogy(SNR, BER_MLNN_100_100, marker='v', label='N1=100, N2=100', color = "red", linestyle='--')

    plt.xlabel('SNR', fontsize=20)
    plt.ylabel('BER', fontsize=20)
    plt.title('Parity(10,5) Multi-label Neural Network BER Estimation', fontsize=20)
    plt.legend([
        'BPSK, Uncoded',
        'Soft-decision ML',
        # 'N=100, Article',
        # 'N=100',
        # 'N1=50, N2=50',
        # 'N1=100, N2=100'
    ], loc='lower left')
    # Display the Plot
    plt.show()

def main():
    # originalplot()
    nnplot()

if __name__ == '__main__':
    main()