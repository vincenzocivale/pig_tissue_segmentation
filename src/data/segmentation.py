import cv2
import numpy as np

def adaptive_thresholding(image, max_value=255, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                          threshold_type=cv2.THRESH_BINARY, block_size=11, C=2):
    """
    Apply adaptive thresholding to an image.

    Parameters:
      - image: input image (grayscale or color).
      - max_value: maximum value assigned to pixels exceeding the threshold (default 255).
      - adaptive_method: adaptive thresholding method; e.g., cv2.ADAPTIVE_THRESH_GAUSSIAN_C or cv2.ADAPTIVE_THRESH_MEAN_C.
      - threshold_type: threshold type; e.g., cv2.THRESH_BINARY or cv2.THRESH_BINARY_INV.
      - block_size: block size (must be odd) for local threshold calculation.
      - C: constant subtracted from the mean or weighted value.

    Returns:
      - thresholded: binary image obtained by applying adaptive thresholding.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if block_size % 2 == 0:
        raise ValueError("block_size must be an odd number.")

    thresholded = cv2.adaptiveThreshold(gray, max_value, adaptive_method,
                                        threshold_type, block_size, C)
    return thresholded


def watershed_segmentation(image):
    # Verifica se l'immagine è in scala di grigi
    if len(image.shape) == 2 or image.shape[2] == 1:
        # Converte l'immagine in BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image

    # Conversione in scala di grigi
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Applicazione della soglia
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    sure_bg = cv2.dilate(thresh, kernel, iterations=3)
    
    # 4. Trasformata della distanza con normalizzazione
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    
    # 5. Rilevamento picchi per foreground sicuro
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Etichettatura dei marcatori
    _, markers = cv2.connectedComponents(sure_fg)

    # Aggiunta di 1 a tutti i marcatori per distinguere lo sfondo dal resto
    markers = markers + 1

    # Etichettatura dell'area sconosciuta con 0
    markers[unknown == 255] = 0

    # Applicazione dell'algoritmo di watershed
    markers = cv2.watershed(image_bgr, markers)

    # Creazione della maschera binaria: oggetti = 255, sfondo = 0
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[markers > 1] = 255

    return mask
