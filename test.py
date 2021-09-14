from neural_network import Network
import json
import sys

# Load network we are using the network from 43rd epoch
network = Network(path='43_small_network.json')
# set default amount to test to all testing data
test_size = 500

# Gets data, tests, and prints the output
def main():
    data, outputs = load_data(test_size, 0)
    dog_correct, cat_correct, dog_guess, cat_guess = test(data, outputs)
    print(f'dogs correct: {dog_correct} cats correct: {cat_correct} dog guesses: {dog_guess} cat guesses: {cat_guess}')

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
        if output > .60:
            output  = 1
        else:
            output = 0
        # Grades output
        if output == outputs[i][0]:
            if output ==  1:
                dog_correct += 1
            else:
                cat_correct += 1
        # Measures dog and cat guesses
        if outputs[i][0] ==  1:
            dogs += 1
        else:
            cats += 1

        if output == 1:
            dog += 1
        else:
            cat += 1
    # Return sprecent correct dog then cat and dog and cat guesses
    return dog_correct/dogs, cat_correct/cats, dog, cat


def load_data(download, place):
    
    dataset = []
    outputs = []
    # Iterates through and loads data
    for i in range(place, place + download):
        # Add cat
        try:
            with open("cat_data/"+ str(i) + ".json") as f:
                img = json.load(f)
                # Make image pixels between 0 and 1
                make_image_decimal(img)
                # Add image and outputs to lists
                dataset.append(img)
                outputs.append([0])
        except:
            pass
        # Add dog
        try:
            with open("dog_data/"+ str(i) + ".json") as f:
                img = json.load(f)
                # Make image pixels between 0 and 1
                make_image_decimal(img)
                # Add image and outputs to lists
                dataset.append(img)
                outputs.append([1])
        except:
            pass
    return dataset, outputs

# Make image pixel values between 0 and 1
def make_image_decimal(img):
    for i in range(len(img)):
        img[i] /= 255
    return img

# Check to test fewer images
if (sys.argv[1] == 'short'):
    test_size = 20
    print('testing the first 40 images')


main()