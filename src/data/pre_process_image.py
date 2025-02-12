from PIL import Image
import numpy as np
import cv2

def load_tif_image(image_path):
    with Image.open(image_path) as img:
        image_array = np.array(img)
    return image_array

def apply_CLAHE(image, clipLimit=2.0, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahe_image = clahe.apply(image)

    return clahe_image

def erode_image(image, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)

def reduce_resolution(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def apply_mask(image, mask):
    mask = (mask > 0).astype(np.uint8) * 255

    if len(image.shape) == 2:
        masked_image = cv2.bitwise_and(image, image, mask=mask)
    else:
        masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image



def apply_sobel(image):
    # Applicazione del filtro Sobel per il rilevamento dei bordi
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Filtro per i bordi lungo l'asse X
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Filtro per i bordi lungo l'asse Y

    # Calcolare la magnitudine dei bordi (combinando le componenti X e Y)
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    
    # Convertire in un'immagine a 8 bit per la visualizzazione
    sobel_edges = cv2.convertScaleAbs(sobel_edges)
    
    return sobel_edges

def bilateral_downsample(image, scale_percent=50, d=9, sigma_color=75, sigma_space=75):
    """
    Applica un filtro bilaterale per preservare i bordi e riduce la risoluzione dell'immagine.

    Parametri:
    - scale_percent (int): Percentuale di riduzione della risoluzione (es. 50 per ridurre del 50%).
    - d (int): Diametro del filtro bilaterale (maggiore = più sfocatura).
    - sigma_color (int): Influenza del colore per il filtraggio (maggiore = più uniforme).
    - sigma_space (int): Influenza della distanza spaziale per il filtraggio.

    Ritorna:
    - downscaled_image (numpy.ndarray): L'immagine ridotta con bordi preservati.
    """

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    # Applica il filtro bilaterale per ridurre il rumore mantenendo i bordi
    filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    # Calcolo delle nuove dimensioni mantenendo le proporzioni
    width = int(filtered_image.shape[1] * scale_percent / 100)
    height = int(filtered_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # Ridimensionamento (downsampling)
    downscaled_image = cv2.resize(filtered_image, dim, interpolation=cv2.INTER_AREA)
    
    return downscaled_image

def enhance_image(img_path, kernel_size=3, iterations=1):
        
    image = load_tif_image(img_path)
    downsampled_image = bilateral_downsample(image, scale_percent=50)
    denoised = cv2.GaussianBlur(downsampled_image, (3,3), 0)
    eroded_image = erode_image(denoised, kernel_size, iterations)
    clahe_image = apply_CLAHE(eroded_image)

    return clahe_image