from neural_network import Network
import json

network = Network(path='30_small_network.json')

def main():
    cats, dogs = load_data(12500, 0)
    print(f'cats {cats} dogs {dogs}')

# Tests the data returns percent correct
def test(data, outputs):
    global network

    # Tests the network
    cat_correct = 0
    dog_correct = 0
    dogs = 0
    cats = 0
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
    # Return percent correct
    return dog_correct/dogs, cat_correct/cats

def load_data(download, place):
    
    dogs = 0
    cats = 0
    # Iterates through and loads data
    for i in range(place, place + download):
        # Add cat
        try:
            with open("cat_data/"+ str(i) + ".json") as f:
                cats += 1
        except:
            pass
        # Add dog
        try:
            with open("dog_data/"+ str(i) + ".json") as f:
                dogs += 1
        except:
            pass
    return cats, dogs

# Make image pixel values between 0 and 1
def make_image_decimal(img):
    for i in range(len(img)):
        img[i] /= 255
    return img

main()