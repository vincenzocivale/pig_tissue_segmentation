import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from PIL import Image

# Funzione per caricare l'immagine
def load_image(image_path):
    img = Image.open(image_path).convert('L')  # Convertiamo in scala di grigi
    img = img.resize((img.width // 2, img.height // 2)) 
    return np.array(img)

# Funzione per aggiornare l'immagine in base alla soglia
def update_image(threshold, image_array):
    binary_image = (image_array > threshold) * 255  # Applica la soglia
    ax.imshow(binary_image, cmap='gray')
    fig.canvas.draw()

# Caricamento dell'immagine (inserisci il percorso corretto)
image_path = 'D:\PIG_slices\BZ1_BZ1_CH1_AUTO_MIP.tif'  # Modifica con il tuo file TIFF
image_array = load_image(image_path)

# Creazione della figura
fig, ax = plt.subplots()
ax.imshow(image_array, cmap='gray')

# Creazione della barra di scorrimento
threshold_slider = widgets.IntSlider(value=128, min=0, max=255, step=1, description='Soglia')

# Collegamento della barra alla funzione di aggiornamento
widgets.interactive(update_image, threshold=threshold_slider, image_array=widgets.fixed(image_array))

# Mostrare i widget
display(threshold_slider)
