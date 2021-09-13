from neural_network import Network

# Create network
network = Network([2, 100, 1])

# Print initial output
print(f'When given [0, 1] the network outputs {network.forward_propagate([0, 1])}')
print(f'When given [1, 0] the network outputs {network.forward_propagate([1, 0])}')

# Train network for 10 epochs
network.train(inputs=[[0,1], [1, 0]], outputs=[[0], [1]], epochs=20)

print('The network was trained to output 0 when the inputs are [0, 1] and output 1 when the inputs are [1, 0]')

# Print final output
print(f'When given [0, 1] the network outputs {network.forward_propagate([0, 1])}, the expected output is 0')
print(f'When given [1, 0] the network outputs {network.forward_propagate([1, 0])} the expected output is 1')