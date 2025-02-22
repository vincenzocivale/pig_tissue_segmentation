import numpy as np
import cv2
from skimage.segmentation import mark_boundaries


def overlay_cluster_mask(image, cluster_map, cluster_index, color=(255, 0, 0), alpha=0.5):
    """
    Sovrappone la maschera di un singolo cluster sull'immagine originale.

    Args:
        image: Immagine in scala di grigi
        cluster_map: Mappa dei cluster
        cluster_index: Indice del cluster di cui si vuole sovrapporre la maschera
        color: Colore (in formato RGB) per evidenziare il cluster (default rosso)
        alpha: Opacità della sovrapposizione (0 è trasparente, 1 è opaco)

    Returns:
        overlayed_image: Immagine con la maschera del cluster sovrapposta
    """
    # Crea una maschera per il cluster specificato
    cluster_mask = (cluster_map == cluster_index).astype(np.uint8)

    # Converti l'immagine originale in un formato a 3 canali per poter sovrapporre il colore
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Crea un'immagine di sovrapposizione con il colore del cluster
    overlay = np.zeros_like(image_rgb)
    overlay[cluster_mask == 1] = color  # Applica il colore solo ai pixel del cluster

    # Sovrapponi la maschera all'immagine originale, regolando l'opacità
    overlayed_image = cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)

    return overlayed_image

def visualize_superpixels_with_boundaries(image, segments):
    """
    Visualizza i superpixel sull'immagine originale, mostrando i bordi dei superpixel.

    Args:
        image: Immagine in scala di grigi
        segments: Mappa dei superpixel

    Returns:
        superpixel_viz: Immagine con i bordi dei superpixel
    """
    return mark_boundaries(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), segments, mode='thick')


