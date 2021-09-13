from neural_network import Network
from test import load_data

# Tests the data returns percent correct
def test(data, outputs):
    global network

    # Tests the network
    cat_correct = 0
    dog_correct = 0
    dogs = 0
    cats = 0
    dog, cat = 0,0
    for i in range(len(data)):
        # Gets output
        output = network.forward_propagate(data[i])[0]
        # Rounds output
        if output > .5:
            output  = 1
        else:
            output = 0
        # Grades output
        if output == outputs[i][0]:
            if output ==  1:
                dog_correct += 1
            else:
                cat_correct += 1
        
        if outputs[i][0] ==  1:
            dogs += 1
        else:
            cats += 1

        if output == 1:
            dog += 1
        else:
            cat += 1
    # Return percent correct
    return dog_correct/dogs, cat_correct/cats, dog, cat

network = Network([2500, 500, 100, 50, 1])

dataset, outputs = load_data(4, 0)

for i in dataset:
    print(network.forward_propagate(i))
print(test(dataset, outputs))

dataset, outputs = load_data(50, 0)

network.train(dataset, outputs, 3)

dataset, outputs = load_data(4, 0)

for i in dataset:
    print(network.forward_propagate(i))
print(test(dataset, outputs))

# dataset, outputs = load_data(50, 0)
# print(test(dataset, outputs))
