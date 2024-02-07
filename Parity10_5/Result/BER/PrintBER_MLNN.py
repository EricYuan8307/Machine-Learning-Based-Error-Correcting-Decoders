import matplotlib.pyplot as plt

# Data Storage Reference:
SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8.0]

BER_uncoded_BPSK = [0.0786522, 0.0668552, 0.0563124, 0.0464074, 0.0374606, 0.029672, 0.0228144, 0.0173094, # 0~3.5
                0.0125062, 0.0087702, 0.00591, 0.003912, 0.002396, 0.0014372, 0.000759, 0.000394, 0.000197# 4.0~8.0
                ]
BER_SDML = [0.0565318, 0.0434518, 0.0326392, 0.0233738, 0.0162642, 0.010814, 0.0068466, 0.0041702, # 0~3.5
        0.0023946, 0.001271, 0.0006384, 0.0002862, 0.0001314, 5.36e-05, 1.81e-05, 6.5e-06, 0.0# 4~8.0
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
    plt.title('Multi-label Neural Network BER Estimation', fontsize=20)
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