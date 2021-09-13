from neuron import Neuron
import json


class Network:
    """
    Forward feeding neural network

    Attributes
    ----------
    network : list
        neural network

    Methods
    -------
    __init__(size_list = [1] , length=0, hidden_layer_size=0, input_output = [1,1], l_rate = .1, path = None)
        Creates the network in a three dimentional list called self.network

        You can make your network three different ways:
            1) You can set size_list to specify the number
            of neurons per layer.
            2) You can set length, hidden_layer_size, and input_output
            to create a network
            3) You can set path to import a network.

    forward_propagate(inputs):
        Forward propagates and returns the outputs of the final row.

    backpropagate(expected_output):
        Backpropagates through the network based on the expected output.

    train(inputs, outputs, epochs):
        Trains the network using a dataset and expected outputs list you provide
        for a specified number of epochs.

    save(path):
        Saves the network in a json file.
    
    
    """


    def __init__(self, size_list = [1] , length=0, hidden_layer_size=0, input_output = [1,1], l_rate = .1, path = None):
        """
        Creates the network in a three dimentional list called self.network

        You can make your network three different ways:
            1) You can set size_list to specify the number
            of neurons per layer.
            2) You can set length, hidden_layer_size, and input_output
            to create a network
            3) You can set path to import a network.

        :param size_list: a list with the number of neurons that should be on 
        row ex. [3, 5, 2, 7, 8] The fist input is the number of inputs and 
        the last entry in the list is the number of output neurons. You can use
        this if you don't want the hidden layers to be one size. 
        :param length: Number of hidden layers.
        :param hidden_layer_size: Hidden layer length
        :param input_output: [number of input neurons, number of output neurons]
        :param path: File path to load network. Set if you want to load.
        :param l_rate: The rate at which the computer will change the weights.
        .1 is often recomended.
        :return: an initilazed neural network
        """
        # Makes a layer of the network
        def make_network(row_length, last_row):
            row = []
            # Add neurons
            for i in range(row_length):
                row.append(Neuron(behind=last_row, l_rate=l_rate))        # last_row is how many weights the neuron should have
            self.network.append(row) 
            return row_length


        self.network = []
        # Loads a network
        if path != None:
            # Load file
            with open(path) as f:
                weights = json.load(f)
            # Assigns weights
            for row in weights:
                layer = []
                for neuron_weights in row:
                    layer.append(Neuron(weights=neuron_weights, l_rate=l_rate))
                self.network.append(layer)
            return

        last_row = 0
        # Uses length paramaters to make new network
        if length != 0:
            # Create first layer
            last_row = make_network(input_output[0], last_row)
            for i in range(length):
                last_row = make_network(hidden_layer_size, last_row)
            # Create last layer
            last_row = make_network(input_output[1], last_row)
            return

        # Create network with list
        for row_length in size_list:
            last_row = make_network(row_length, last_row)
    
    def forward_propagate(self, inputs):
        """
        Forward propagates and returns the outputs of the final row

        :param inputs: the values you want assigned to the first
        layer of neurons
        :return: the final layer of neurons
        """

        # Defaults all of the unentered inputs to 0 so you do not
        # have to provide as many inputs as you have input neurons.
        for i in range(len(self.network[0]) - len(inputs)):
            inputs.append(0)

        # Assigns inputs
        for i in range(len(inputs)):
            self.network[0][i].val = inputs[i]
        # Makes the values of the last row the values of the inputs
        last_row = inputs
        # Add bias
        last_row.append(1)
        # Interates through network calling the activation function
        # of each neuron and recording the output (neuron.val) of each neuron
        # to be passed into the next layer (row)
        for i in range(1, len(self.network)):
            row = self.network[i]
            current_row = []

            for neuron in row:
                neuron.activate(last_row)       # last_row is the outputs (val) of the last row
                # Record the outputs in current_row to become last_row
                current_row.append(neuron.val)
            # Switch rows
            last_row = current_row
            # Add bias
            last_row.append(1)
        # Remove bias from inputs so it can be reused
        del inputs[-1]
        # Return current_row minus the bias
        return current_row[:-1]
    
    def backpropagate(self, expected_output):
        """
        Backpropagates through the network based on the expected output.

        :param expected_output: A list of the expected outputs
        """

        # Gets the values/ outputs of the neurons network[positon]
        # retuns list
        def get_inputs(position):
            inputs = []
            for input_neuron in self.network[position]:
                inputs.append(input_neuron.val)
            # Add bias
            inputs.append(1)
            return inputs

        # Backpropagate the output layer
        output_layer = self.network[-1]
        # Get outputs of last layer
        inputs = get_inputs(-2)
        
        for i in range(len(output_layer)):
            # Get weights before update
            output_layer[i].output_layer_backpropagate(expected_output[i], inputs)


        # Reverse iterate through the middle on the network and backwardpropagate. 
        # We don't do the input layer or the output layer because that would cause an error,
        # since they are not surrounded by other neurons. We don't have to backpropagate
        # the input layer but we need to do it to the output layer which we did above.
        # Cutting out the input layer causes i to be one below the index we want. That
        # is dealt with below.
        for i in reversed(range(len(self.network[1:-1]))):
            i += 1      # Move up one index
            # Set up arguments for backpropogation function in this layer
            inputs = get_inputs(i-1)
            layer = self.network[i]
            forward_layer = self.network[i + 1]
            # Get errors from neurons in forward layer
            forward_errors = []
            forward_weights = []
            for neuron in forward_layer:
                forward_weights.append(neuron.weights.copy())
                forward_errors.append(neuron.error)
            # Backpropagate neurons in layer
            for i in range(len(layer)):
                layer[i].backpropagate(forward_errors, forward_weights, i, inputs)

    def train(self, inputs, outputs, epochs):
        """
        Trains the network using a dataset and expected outputs list you provide
        for a specified number of epochs.

        :param inputs: A list containing all of the input lists for training.
        :param outputs: A list containing the expected outputs lists
        corosponding to inputs. Basically outputs[i]
        is the expected output for inputs[i].
        :return: A list of the average losses of each epoch.
        This is done by rounding to one or zero so if you
        expect outputs that are between 1 and zero this number
        is meaningless.
        """
        # Trains one epoch returns loss
        def train_one_epoch(inputs, outputs):
            losses = 0
            for i in range(len(inputs)):
                
                # forward and backpropagates one image
                real_output = self.forward_propagate(inputs[i])
                self.backpropagate(outputs[i])
                # Save loss
                for j in range(len(real_output)):
                    if real_output[j] > .5:
                        real_output[j] = 1
                    else:
                        real_output[j] = 0
                    if real_output[j] != outputs[i][j]:
                        losses += 1
                        break
            # return and calculate loss
            return losses/len(inputs)

        # List of losses per epoch
        losses = []
        print("Training")
        # Indent to add to all terminal outputs
        indent = "\t"
        for i in range(epochs):
            # Outputs data to the user, trains one epoch, and saves loss
            loss = train_one_epoch(inputs, outputs)
            print(f"{indent}epoch: {i + 1}/{epochs} loss = {loss}")
            losses.append(loss)
        return losses
            

    def save(self, path):
        """
        Saves the network in a json file.

        :param path: The path to the file you to save this network in.
        """
        # Create weights list
        weights = []
        for row in self.network:
            layer = []
            for neuron in row:
                layer.append(neuron.weights)
            weights.append(layer)
        # Save weights
        with open(path, "w") as f:
            json.dump(weights, f)
