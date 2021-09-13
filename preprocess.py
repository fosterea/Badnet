from neural_network import Network
from PIL import Image
from numpy import asarray
import json

# Image dimensions
img_width = 50

# Loads one image
def load_image(path):
    # Open with PIL Image and convert to grayscale
    with Image.open(path).convert('L') as image:
        global img_width
        # Resize image
        image = image.resize((img_width, img_width))
        # Flatten image and turn into list
        image = asarray(image)
        image = image.flatten().tolist()
        return image

# Returns 2D list of images.
def get_dataset():
    dogs = []
    cats = []
    # Loads images
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

# Loads and Saves the images as Json files.
def download_images():
    # Loads images
    dogs, cats = get_dataset()
    # Saves images
    for i in range(len(dogs)):
        with open("dog_data/" + str(i) + ".json", "w") as outfile:
            json.dump(dogs[i], outfile)
    for i in range(len(cats)):
        with open("cat_data/"+ str(i) + ".json", "w") as outfile:
            json.dump(cats[i], outfile)