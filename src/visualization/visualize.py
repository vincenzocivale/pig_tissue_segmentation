import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from PIL import Image
import cv2


# Funzione per caricare l'immagine
def load_image(image_path):
    img = Image.open(image_path).convert('L')  # Convertiamo in scala di grigi
    img = img.resize((img.width // 2, img.height // 2)) 
    return np.array(img)



# Funzione per plottare l'istogramma dell'immagine
def plot_histogram(image_array):
    plt.figure()
    plt.hist(image_array.ravel(), bins=256, range=(0, 256), density=True)
    plt.title('Histogram')
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')
    plt.show()

# Funzione per plottare l'immagine e attendere la chiusura della finestra
def plot_image(image_array):
        plt.figure()
        plt.imshow(image_array, cmap='gray')
        plt.title('Image')
        plt.axis('off')
        plt.show(block=True)

