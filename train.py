from neural_network import Network
import json

# Trains the network
def main():
    # Loads the network
    global network
    network = Network(path="small_network.json")
    # Size of the testing data
    test_size = 1000
    # Adjusts for each number downloads two images
    test_size /= 2
    test_size = int(test_size)
    # Downloads 100 cats and 100 dogs at a time
    download = 100
    # Gets a dictionary of all the losses
    with open("stats.json") as f:
        stats = json.load(f)
    # Gets a dictionary of the place in the training process
    with open("place.txt") as f:
        start = int(f.readline())
        rotations = int(f.readline())
        epoch_start = int(f.readline())
    # Sets n epochs
    epochs = 3
    # Iterates through epochs
    for i in range(epoch_start, epochs):
        # Get list to save losses 
        if not f"{i}th_losses" in stats:
            stats[f"{i}th_losses"] = []
        # Iterates though pictures
        while(start < 12500):
            # Save place
            try:
                with open("place.txt", "w") as f:
                    f.write(f"{start}\n{rotations}\n{i}")
            except:
                print("We were unable to record the starting position. Go to place.txt and correct the "
                 + "value to " + start)
            # Print epoch
            print(f"epoch {i}")
            # Load data
            print("Loading data")
            dataset, outputs = load_data(download, start)

            # Trains network and adds loss
            stats[f"{i}th_losses"].append(network.train(dataset, outputs, 1))
            with open("stats.json", "w") as f:
                json.dump(stats, f)
            # Save network
            print("Saving")
            network.save("network.json")
            # Iterate traking variables 
            start += download
            rotations += 1
        # Reset traking variables
        rotations = 1
        start = test_size
        

# Imports the images from the json files
# Return dataset, outputs
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


main()