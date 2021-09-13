import random
import math


class Neuron:
    """
    Neuron in a forward feeding neural network

    Attributes
    ----------
    l_rate : float
        Learning rate
    weights : list[float]
        Weights for the neurons in the previous layer.
    n_rear : int
        Number of neurons in the previous layer.
    val : float
        The "value" of the neuron. Does not exist before activate is called.
    error : float
        The error of the neuron. Does not exist before backpropagate or 
        output_layer_backpropagate is called.

    Methods
    -------
    __init__(behind=0, weights=None, l_rate=.1):
        Inits a neuron with random weights if you set behind,
        and inits a neuron with specified weights if you set
        weights.

    activate(last_row):
        Calculates the value of the neuron. 

    backpropagate(forward_errors, forward_row, ith_place, inputs):
        Backpropagates one neuron, updating the error of the neuron and updating
        the weights.

    output_layer_backpropagate(expected, inputs):
        Backpropagates an output layer neuron.
    

    """
    
    def __init__(self, behind=0, weights=None, l_rate=.1):
        """
        Inits a neuron with random weights if you set behind,
        and inits a neuron with specified weights if you set
        weights. 


        :param behind: the number of neurons in the layer 
        behind this layer (not incuding bias)
        :param weights: Optionally sets the weights. No consideration
        is taken from behind.
        :param l_rate: The rate at which the neuron should update its
        weights based on the errors.
        :return: an initialized neuron
        """
        # Save l_rate
        self.l_rate = l_rate
        # Sets weights if they were provided
        if weights != None:
            self.weights = weights
            for i in range(len(weights)):
                behind += 1
            self.n_rear = behind
            return
        
        # Sets number of neurons in previous row + 1 for bias
        self.n_rear = behind + 1
        rand = random.Random()  # Gets Random instance
        self.weights = list()
        # Sets weights
        for i in range(self.n_rear):
            self.weights.append(rand.random() * rand.choice((-1, 1)))

        
    def activate(self, last_row):
        """
        Calculates the value of the neuron. 
        
        :param last_row (list): a list of the values/outputs of the neurons
        from the previous layer + the bias value
        """
        self.val = 0
        # Calculates value based on weights
        for i in range(self.n_rear):
            self.val += self.weights[i] * last_row[i]
        # Sigmoid function to make 1 > val > 0
        try:
            self.val = 1.0 / (1.0 + math.exp(-self.val))
        # Deal with overflow erros realated to giving
        # math.exp a high absolute value
        except:
            if self.val > 0:
                self.val = 1
            else:
                self.val = 0

    # Helper function that adjusts the error for slop and updates weights
    def _adjust(self, inputs):
        l_rate = self.l_rate
        # Adjust the error for slope
        self.error *= self.val * (1.0 - self.val)
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] += inputs[i] * self.error * l_rate

    def backpropagate(self, forward_errors, all_forward_weights, ith_place, inputs):
        """
        Backpropagates one neuron, updating the error of the neuron and updating
        the weights.
        
        :param forward_errors: list of the errors from the layer in front of this one
        :param all_forward_weights: The weights of neurons ahead of this one.
        :param ith_place: The position this neuron is in the layer and the position it
        would be in the weights of the neurons in the next layer.
        :param inputs: The values of the neurons in the layer behind this one.
        """

        # Finds the weights the neurons ahead of this one have for it.
        forward_weights = []
        for weights in all_forward_weights:
            forward_weights.append(weights[ith_place])

        self.error = 0
        # Calculate the error
        for i in range(len(forward_errors)):
            self.error += forward_errors[i] * forward_weights[i]
        # Adjusts the error for slope then adjusts the weights
        self._adjust(inputs)
        

    def output_layer_backpropagate(self, expected, inputs):
        """
        Backpropagates an output layer neuron.

        :param expected: The expected output
        :param inputs: The values of the neurons in the layer behind this one.
        """
        # Calculate the error
        self.error = expected - self.val
        # Adjusts the error for slope then adjusts the weights
        self._adjust(inputs)

