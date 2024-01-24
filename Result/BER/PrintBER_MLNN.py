import matplotlib.pyplot as plt

# Data Storage Reference:
SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5]

article_BPSK = [0.08, 0.07, 0.057, 0.047, 0.04, 0.0305, 0.023, 0.019, # 0~3.5
                0.014, 0.009, 0.0062, 0.004, 0.0025, 0.0016,
                0.0007, 0.00041,
                # 0.0002, 0, 0, 0, 0
                ]
article_SDML = [0.08, 0.065, 0.059, 0.04, 0.029, 0.0205, 0.014, 0.008, 0.0055, # 0~4
        0.0031, 0.00175, 8.1e-04, 3.9e-04, 1.6e-04, 5.5e-05, 1.4e-05, # 4.5~7.5
        # 0, 0, 0, 0, 0
        ]


uncoded_BPSK = [0.07864588, 0.0670532, 0.05630509, 0.046395885, 0.03750538, 0.02963852, 0.02287882, 0.01716848, # 0~3.5
                0.012505335, 0.00879166, 0.00595442, 0.003866035, 0.00238995, 0.00140195, # 4~6.5
                0.00076981, 0.000398745,
                # 0.000193015, 8.405e-05, 3.335e-05, 1.21e-05, 3.855e-06
                ]
SDML = [0.082579625, 0.0665316, 0.05207925, 0.039591475, 0.028990875, 0.020399225, 0.013703275, 0.008746775, # 0~3.5
        0.0052823, 0.00298175, 0.001558025, 0.000787275, 0.000358275, 0.0001452, # 4~6.5
        5.7725e-05, 1.9325e-05,
        # 5.525e-06, 1.417e-06, 0, 0, 0
        ]

MLNN = [0, 0, 0, 0, 0, 0, 0, 0, # 0~3.5
        0, 0, 0, 0, 0, 0, # 4~6.5
        0, 0]


def nnplot():
    plt.figure(figsize=(10, 10))
    plt.semilogy(SNR, uncoded_BPSK, marker='.', label='BPSK, Uncoded')
    plt.semilogy(SNR, SDML, marker='x', label='Soft-decision ML')
    plt.semilogy(SNR, article_BPSK, marker='.', label='BPSK, Uncoded, Article')
    plt.semilogy(SNR, article_SDML, marker='x', label='Soft-decision ML, Article')
    # plt.semilogy(SNR, MLNN, marker='.', label='Multi-label Neural Network')

    plt.xlabel('SNR')
    plt.ylabel('BER')
    plt.title('BER Estimation')
    # plt.legend(['uncoded_BPSK', 'SDML', 'MLNN'], loc='lower left')
    plt.legend(['BPSK, Uncoded', 'Soft-decision ML', 'BPSK, Uncoded, Article', 'Soft-decision ML, Article', 'Multi-label Neural Network'], loc='lower left')

    # Display the Plot
    plt.show()

def main():
    # originalplot()
    nnplot()

if __name__ == '__main__':
    main()