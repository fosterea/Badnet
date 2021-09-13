from neural_network import Network
from PIL import Image
from numpy import asarray
import json

network = None
test_data, test_outputs, train_data, train_outputs = None, None, None, None
download = 50
test_size = 10
img_width = 50
input_size = img_width * img_width

key = {
    "cat" : 0,
    "dog" : 1,
    1 : "dog",
    0 : "cat",
}

def main():
    global test_data
    global test_outputs
    global train_data
    global train_outputs
    global network

    command = input("How would you like to get your network? (load or new): ")
    while True:
        if command == "load":
            load()
        elif command == "new":
            new()
        elif command == "end":
            answer = input("Are you sure you want to end this program if you have not saved your network? (end/cancel) ")
            if answer == "end":
                return
        #elif command == "use":
            #use()
        elif command == save:
            save()
        elif command == "train":
            train()
        elif command == "test":
            import_data()
            print(f"Training data {test(train_data, train_outputs)}")
            #print(f"Testing data {test(test_data, test_outputs)}")
        elif command == "settings":
            global download
            global test_size
            download = int(input("Download: "))
            test_size = int(input("Test size: "))
            test_data = None
            import_data()
        command = input("\nChoose a command load, new, train, use, save, test, settings, or end: ")

    dataset, outputs = load_data()

    network = Network(size_list=[75*75, 100, 100, 100, 1])
    print(str(network.forward_propagate(dataset[0])) + " " + str(outputs[0]))
    print(str(network.forward_propagate(dataset[1])) + " " + str(outputs[1]))
    network.train(dataset, outputs, 2)
    print(str(network.forward_propagate(dataset[0])) + " " + str(outputs[0]))
    print(str(network.forward_propagate(dataset[1])) + " " + str(outputs[1]))
    network.save("network.json")
    network = Network(path="network.json")
    print(str(network.forward_propagate(dataset[0])) + " " + str(outputs[0]))
    print(str(network.forward_propagate(dataset[1])) + " " + str(outputs[1]))

def load():
    global network
    path = input("File Pathway: ")
    l_rate = float(input("Which learn rate do you want to use? .1 is usually recommended "))
    network = Network(path=path, l_rate=l_rate)
    print("Import complete!")

def new():
    global network
    l_rate = float(input("Which learn rate do you want to use? .1 is usually recommended "))
    length = int(input("How many hidden layers should your network have? "))
    hidden_layer_size = int(input("Hidden layer size: "))
    network = Network(length=length, hidden_layer_size=hidden_layer_size, input_output=[input_size, 1] ,l_rate=l_rate)

def use():
    global network
    path = input("File path")

def train():
    global test_data
    global test_outputs
    global train_data
    global train_outputs
    global network
    import_data()
    epochs = int(input("Epochs: "))
    network.train(train_data, train_outputs, epochs)
    #print(f"Training data {test(train_data, train_outputs)}")
    print(f"Testing data {test(test_data, test_outputs)}")

def save():
    #path = input("Path: ")
    path = "network.json"
    global network
    network.save(path)



def test(data, outputs):
    global network
    correct = 0
    for i in range(len(data)):
        output = network.forward_propagate(data[i])[0]
        if output > .5:
            output  = 1
        else:
            output = 0
        if output == outputs[i][0]:
            correct += 1
    return correct/len(data)


def load_image(path):
    with Image.open(path).convert('L') as image:
        global img_width
        image = image.resize((img_width, img_width))
        image = asarray(image)
        image = image.flatten().tolist()
        return image

def get_dataset():
    dogs = []
    cats = []
    for i in range(12500):
        try:
            cats.append(load_image("Dataset/Cat/" + str(i) + ".jpg"))
        except:
            pass
        try:
            dogs.append(load_image("Dataset/Dog/" + str(i) + ".jpg"))
        except:
            pass

    return dogs, cats


def download_images():
    dogs, cats = get_dataset()

    for i in range(len(dogs)):
        with open("dog_data/" + str(i) + ".json", "w") as outfile:
            json.dump(dogs[i], outfile)
    for i in range(len(cats)):
        with open("cat_data/"+ str(i) + ".json", "w") as outfile:
            json.dump(cats[i], outfile)

def load_data():
    global download
    
    dataset = []
    outputs = []
    for i in range(download):
        try:
            with open("cat_data/"+ str(i) + ".json") as f:
                img = json.load(f)
                make_image_decimal(img)
                dataset.append(img)
                outputs.append([0])
        except:
            pass
        try:
            with open("dog_data/"+ str(i) + ".json") as f:
                img = json.load(f)
                make_image_decimal(img)
                dataset.append(img)
                outputs.append([1])
        except:
            pass
    return dataset, outputs

def make_image_decimal(img):
    for i in range(len(img)):
        img[i] /= 255
    return img

def import_data():
    global test_data
    global test_outputs
    global train_data
    global train_outputs
    global network
    global test_size
    if test_data == None:
        print("Loading data")
        dataset, outputs = load_data()
        test_data, test_outputs = dataset[:test_size], outputs[:test_size]        
        train_data, train_outputs = dataset[test_size:], outputs[test_size:]

main()
print(len(network.network))