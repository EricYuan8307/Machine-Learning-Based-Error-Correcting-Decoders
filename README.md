# Master Thesis: Improving the Decoding of Error Correction Codes via Machine Learning

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Installation

Instructions on how to install the program.

### Prerequisites
- python
- torch
- numpy
- matplotlib
- galois

### Step-by-Step Installation

1. Clone the repository:
   ```bash
   git@github.com:EricYuan8307/master-thesis.git

## Function Usage:

### Usage of `generating.py`
`generating.py` is a Python script that generate the codebook for specific length codeword used in Maximum Likelihood.

### Usage of Encode file:
Encode file contains all files in transmitter including `Encoder.py`, `Generator.py` and `Modulator.py`. 

### Usage of Decode file:
Decode file contains all files in receiver side:
- `Converter.py`: decimal to binary for SLNN and hard-decision for MLNN.
- `Decoder.py`: multiply with H matrix.
- `HardDecision.py`: hard decision for Hard-decision maximum likelihood and belief propagation.
- `LDPC_BP.py`: Belief propagation.
- `likelihood.py`: calcuate log-likelihood for belief propagation.
- `MaximumLikelihood.py`: Soft-decision and hard-decision maximum likelihood.
- `NNDecoder.py`: Model for SLNN, new SLNN, MLNN, Optimzied MLNN.

### Usage of Hamming74 file:
Used for SLNN post-pruning. For post-pruning, it needs mask to mute specific edges not to be trained. 
- `BLER_SLNN7_Decrease_Portion.py`: In the end of pruning, several mask to delete specific edge one by one.
- `BLER_SLNN7_Portion.py`: In order to present the pruning usage, I deleted the edges in hidden layer from 0 to 183 
in Hamming (7,4) SLNN, N=7. the portion shows how the removed edges affect the BLER.
- `BLER_SLNN_Optimize.py`: Designed for Parity(26,10) and more different FEC encoding method.
- `BLER_SLNN_Portion.py`: Same as `BLER_SLNN7_Portion.py`, but for Parity(26,10) and more different FEC encoding method.
- `mask.py`: specific mask for Hamming(7,4) pruning.
- `SLNN7_decrase.py`: For Hamming(7,4) after pruning model training.

### Usage for `Errorrage.py` in Metrics file:
Calcuate Bit Error Rate (BER) and Block Error rate (BLER)

### Usage for Result file:
save BER/BLER and model.

### Usage for Transformer file:
- `Functions.py`: FEC code functions for Transformer-based decoder.
- `model.py`: ECCT model for error correction code.
- `Model_ca.py`: CrossMPY for error correction code.

### Usage for Transmit file:
- `noise.py`: BI-AWGN channel.
- `NoiseMeasure.py`: Measure the noise with Eb/No and Es/N0.

### In main file:
- `CrossMPT_estimate.py`: 
Estimate performance of CrossMPT.

- `CrossMPT_HD.py`: 
Hard input CrossMPT for testing.

- `CA_Transformer_main.py`: 
CrossMPT training.

- `earlystopping.py`: 
early stopping used in NN decoder for enshort time of training.

- `ECCT_estimate.py`:
Estimate performance of ECCT.

- `ECCT_HD.py`: 
Hard input ECCT for testing.

- `ECCT_Training.py`: 
ECCT training.

- `ECCT_reduce.py`: 
ECCT training with reduced input, no parity bits input.

- `Edge_detect.py`:
Find the edge in the pruned model.

- `metric_estimation.py`:
Get metric estimation for the FEC encoding method including Uncoded BPSK, HDML, SDML and BP.

- `MLNN_estimate.py`:
Estimate performance of MLNN.

- `MLNN_Training.py`: 
MLNN training.

- `model_threshold.py`: 
For pruning, delete the edges with the smallest portion of weights.

- `NN_decrease.py`:
SLNN for egde deletion and save as new model.

- `NN_estimate.py`:
To estimate all kinds of NN decoder.

- `NN_Optimize.py`:
For the pruned NN decoder, task for training.

- `NN_Protion.py`:
For the pruned NN decoder, the edges are deleted with specific number, get the result for all the saved models.

- `NNdecoder_Training.py`:
Training for all NN decoder including SLNN, new SLNN, MLNN, Optimzied MLNN.

## References:
### Implement the Belief propagation:
Belief Propagation Reference: <br>
D. E. Hocevar, "A reduced complexity decoder architecture via layered decoding of LDPC codes," IEEE Workshop on Signal Processing Systems, 2004. SIPS 2004., Austin, TX, USA, 2004, pp. 107-112, doi: 10.1109/SIPS.2004.1363033.

### Implement Neural Network Decoder:
Neural Network Decoder Reference: <br>
C. T. Leung, R. V. Bhat and M. Motani, "Low Latency Energy-Efficient Neural Decoders for Block Codes," in IEEE Transactions on Green Communications and Networking, vol. 7, no. 2, pp. 680-691, June 2023, doi: 10.1109/TGCN.2022.3222961.

### Error Correction Code Transformer:
Transformer Reference: <br>
Yoni Choukroun and Lior Wolf. Error correction code transformer. arXiv preprint arXiv:2203.14966, 2022.

### Cross-attention Transformer for Error Correcting Codes:
Cross-attention Transformer Reference: <br>
Seong-Joon Park and Hee-Youl Kwak and Sang-Hyo Kim and Yongjune Kim and Jong-Seon No. CrossMPT: Cross-attention Message-Passing Transformer for Error Correcting Codes. arXiv preprint arXiv:2405.01033, 2024

To train the model faster, I used early stopping.

### Addition: In Good First Issue, I mentioned some common bugs and solutions in Belief Pragation, BER and BLER.


