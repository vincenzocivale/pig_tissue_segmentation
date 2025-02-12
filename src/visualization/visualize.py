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


# Caricamento dell'immagine (inserisci il percorso corretto)
image_path = r'C:\Users\cical\Documents\GitHub\Repositories\pig_tissue_segmentation\equalized_image.jpg' # Modifica con il tuo file TIFF
image_array = load_image(image_path)


# Funzione per plottare l'istogramma dell'immagine
def plot_histogram(image_array):
    plt.figure()
    plt.hist(image_array.ravel(), bins=256, range=(0, 256), density=True)
    plt.title('Histogram')
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')
    plt.show()

# Plot dell'istogramma dell'immagine caricata
plot_histogram(image_array)
