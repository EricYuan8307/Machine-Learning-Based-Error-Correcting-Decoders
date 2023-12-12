import torch
import torch.nn as nn
import torch.nn.functional as F

class SpikingNeuronLayer(nn.Module):
    def __init__(self, num_inputs, num_neurons):
        super(SpikingNeuronLayer, self).__init__()
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons

        # Weight matrix for connections between input neurons and output neurons
        self.weights = nn.Parameter(torch.rand(num_neurons, num_inputs))

        # Membrane potential threshold for spike generation
        self.threshold = 1.0

        # Decay factor for the membrane potential
        self.decay = 0.9

        # Simulation time step
        self.dt = 0.1

    def forward(self, x):
        # Initialize membrane potentials and spikes
        membrane_potential = torch.zeros(self.num_neurons)
        spikes = torch.zeros(self.num_neurons)

        # Simulation loop
        for t in range(x.size(1)):
            # Update membrane potential using input and decay
            membrane_potential = self.decay * membrane_potential + torch.mm(self.weights, x[:, t, :].t())

            # Check for spike generation
            spike_mask = (membrane_potential >= self.threshold).float()
            spikes += spike_mask

            # Reset membrane potential after spike
            membrane_potential = membrane_potential * (1 - spike_mask)

        # Output is the spike count for each neuron
        return spikes

# Create an instance of the SpikingNeuronLayer
snn_layer = SpikingNeuronLayer(num_inputs=2, num_neurons=1)

# Input spike trains (batch_size, time_steps, num_inputs)
input_spikes = torch.randint(0, 2, size=(3, 10, 2)).float()

# Forward pass through the SNN layer
output_spikes = snn_layer(input_spikes)

# Print the input and output spike trains
print("Input Spikes:", input_spikes)
print("\nOutput Spikes:", output_spikes)
