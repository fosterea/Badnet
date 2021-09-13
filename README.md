# Neural Network
#### Video Demo: <https://share.vidyard.com/watch/9g5Yem8GrEJEkz1d4iemDM?>
#### Description:

To run any of the programs in this directory you must use python 3.

The Neural Network directory contains both the files required to make your own forward feeding neural network as well as a few files dedicated to demonstrating how to use this “library.” You can read more about neural_network.py, neuron.py, and how to make your own neural network in the Network and Neuron section. And you can how I made a neural network in the demo section.

# [Network](neural_network.py)

The Network class from [neural_network.py](neural_network.py) represents a forward feeding, backpropagating neural network. This network will only output values between 0 and 1 because
it uses a sigmoid function to flatten the neuron values. There is docstring documentation of the Network class.


### Initialize
To create your network, you can use several different sets of parameters:

1) You can set provide a list specifying the number of neurons per layer:
    ```    
    network = Network(size_list=[5, 10, 8, 2])
    ```      
2) You can set length, hidden_layer_size, and input_output to create a network without
having to provide the size of every layer:
    ```        
    network = Network(length=3, hidden_layer_size=8, input_output=[2, 1])
    ```     
3) You can import a network from a JSON file:
    ```  
    network = Network(path='network.json')
    ```    
### Forward Propagate
A Network object contains a 2D list (network) of Neuron objects. The network
outputs values by iterating through each layer then each neuron and calculating its value
based on the previous layer. Each neuron has a list of weights corresponding to the neurons
in the last layer. The value of the current neuron is the sum of the values of neurons
in the previous layer times the corresponding weights. This process is repeated
for every neuron in the network until you get the values of the output neurons. This process
is called forward propagation and can be used by calling `forward_propagate`
and providing the values of the input neurons. The forward propagation function returns a list of the values of the neurons in the output layer. Example:
```
inputs = [.5, 1]
output = network.forward_propagate(inputs)
``` 

### Backpropagate
To make your network more accurate, you use a process called backpropagation calling
`backpropagate` and providing the expected outputs. The final values
of your output are still stored in your network, so you don’t have to provide them. The network then subtracts the expected
output values from the actual output values creating the error. The network then adjusts
the weights by the value of the weight * the error. The next layer of neurons calculates their errors by adding up the amount other neurons' weights got adjusted. The network continues this
process until all weights have been adjusted. By calling this function, you will make the actual
output more like the expected output. Example:
```
expected_outputs = [1, 0]
network.backpropagate(expected_outputs)
```
### Train
The train function automates the training process by completing the forward and backpropagation processes for you. To use `train`, you must provide a 2D list of all the inputs and a corresponding 2D list of all the expected outputs. The final parameter is epochs/training periods, the number of repetitions you want your network to iterate over your training data. The train function also rounds the outputs to either 1 or 0, then checks if they are equal to the expected output. If your expected output is exclusively ones and zeros, train will return a list of losses by epoch. Example:
```
inputs = [[0, 1, 0.75], [1, 0, 0]]
outputs = [[0, 1], [1, 1]]
epochs = 3
losses = network.train(inputs, outputs, epochs)
```

### Save
After you have trained your network, you are going to want to save it for later use. To do this
Call the save function and provide a file path. This will save the weights of your network in a JSON file, which can be loaded when you initialize a new network. Example:
```
path = ‘network.json’
network.save(path)
```

# [Neuron](neuron.py)

The Neuron class is imported from [neuron.py](neuron.py) by the Network class to automate many functions related to a specific neuron. This just made it easier mentally to write and debug Network.

### These functions are:

- `__init__`: Inits a neuron with random weights if you set behind,
        and inits a neuron with specified weights if you set
        weights. 
- `activate`: Calculates the value of the neuron based on the previous layer. Uses sigmoid function. 
- `backpropagate`: Backpropagates one neuron, updating the error of the neuron and updating
        the weights.
- `output_layer_backpropagate`: Bacpropagates an output layer neuron, calulating the error and updating the weights based on the expected output and the actual output.

If you are want to learn more about how to use the Neuron class there are comments and docstring.
# Demo

### Procuring data

To make my neural network, I first had to obtain training and testing data. I used the top result on Kaggle: <https://www.kaggle.com/shaunthesheep/microsoft-catsvsdogs-dataset> This dataset contains 25,000 images of cats and dogs, split 50 50. Because the dataset is so large, I only uploaded the preprocessed JSON files. The first 500 cats and the first 500 dogs are the testing data.

I used the Numpy and PIL libraries to convert my pictures to a grayscale 75 by 75 1D list to feed my network. Because this process was slow and only needed to be done once, I decided to preprocess the dataset and store the images as individual JSON files. You can see this code in [preprocess.py](preprocess.py).

### Training

 I used [train.py](train.py) to iterate over the data in chunks of 100 cats and 100 dogs to train my networks and then save it so as not to lose any progress. This program used [place.txt](place.txt) to keep track of the current position, so if the program were interrupted by me or the computer crashing, I would not lose all the progress. This allowed me to update code in the middle of Training my network.

I first began with a large network with 5,625 or 75 * 75 input neurons and eight hidden layers of 1,000 neurons. This network took an extremely long time to get through each epoch, so I decided to use a much smaller network. I first decided to shrink the images from 75 by 75 to 50 by 50 pixels requiring a second preprocessing. The small network was created by setting the size_list parameter to [2500 (50 * 50), 500, 200, 100, 50, 1]. (At this point, I am randomly guessing at the correct number of neurons and hidden layers.) 

Although big brains are generally better than small brains, this only holds true if the big brains can use all their neurons efficiently. Sadly this is not the case with my neural network because it needs data and time to figure out how to use its neurons effectively. I was not about to spend months training this network, and even if I did, all that time could have led to the network over specializing in the last 24,000 pictures from the first Kaggle result when you search cats vs. dogs. Although the old network's ability to consider more possibilities, a smaller network is better in this case.

### Bias

So there is no easy way to say this, but Nety is anti-cat. When I tested for bias, I noticed the network was right about dogs much more often than about cats. So my first thought was the network library’s fault. However, after conducting a few tests, I think that cats are inherently flawed, it is Microsoft’s fault (they made the dataset), my preprocessing may make cats less distinguishable than dogs, or it is still my fault, and I haven’t found the problem. Your choice. 

I tried making a few new networks and training them on a small amount of data for a few epochs. When I trained them, I reversed my enumeration scheme from 1 for dog and 0 for cat to 0 for dog and 1 for cat. The network still gave me biased results, and I ruled out that it had to do with the error calculation system. 

I still don’t know why there were biased results, but I could reverse the trend by saying the correct result for cat is -0.01 instead of 0. This was to make the error calculation system think the network was farther off when it came to cats than it really was. I overshot this process and made the network biased against dogs. Luckily I made a copy of the network earlier in this process which is still biased against cats. At this point, I want to move on to CS50’s AI course, so I am going to just use this network instead of spending more time training the network more. (I have spent at least a month on this project)

While I was testing for bias, I realized that I could change my rounding process to make the output one if the output was greater than .6 or .75 instead of .5. This would give cats an advantage and lead to less bias. With the current network and testing data, when I round up or down at .5, the network is correct about dogs 73.4% of the time and cats 53.8% of the time; there are 598 dog guesses and 402 cat guesses. This is better than the low point when the network was only correct about cats around 35% of the time. Now when I round at 0.6, the bias slightly reverses: 
```
dogs correct: 0.646, cats correct: 0.66, dog guesses: 493, cat guesses: 507
```
This works out to an average accuracy of 65.3% with the testing data.

### Use

To use this passing network, you can run [test.py](test.py) without any arguments to test all the testing data, but it will take you over 10 minutes, or you can test the first 40 images by running `python3 test.py short`.

To see a much shorter example of the neural network library, you can use [demo.py](demo.py) calling `python3 demo.py`. This file creates and trains a neural network for 20 epochs. It will be self-explanatory. I used programs like this to test if Network was working.
