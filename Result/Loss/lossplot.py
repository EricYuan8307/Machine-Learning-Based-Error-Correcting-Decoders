import os
import json
import matplotlib.pyplot as plt

def plotLoss(loss_data_file):
    # Save the loss data to the specified JSON file
    with open(loss_data_file, 'w') as f:
        json.dump(loss_data, f)

    # Extract the training and testing loss lists
    train_losses = loss_data['train_losses']
    test_losses = loss_data['test_losses']

    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Testing Loss', marker='o')
    plt.xlabel('MLNN Epoch')
    plt.ylabel('MLNN Loss')
    plt.title('MLNN Training and Testing Loss')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

plotLoss("Loss", "MLNN_loss_data_SNR0.0_2024-01-22_13-56-11.json")