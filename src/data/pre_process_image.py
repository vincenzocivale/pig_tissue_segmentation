from PIL import Image
import numpy as np
import cv2
from skimage import transform
from skimage.segmentation import slic

Image.MAX_IMAGE_PIXELS = None

def load_tif_image(image_path):
    with Image.open(image_path) as img:
        image_array = np.array(img)
    
    # Normalizza se necessario
    if image_array.dtype != np.uint8:
        image_array = (255 * (image_array / image_array.max())).astype(np.uint8)
    
    return image_array


def apply_CLAHE(image, clipLimit=2.0, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahe_image = clahe.apply(image)

    return clahe_image

def erode_image(image, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)



def apply_mask(image, mask):
    # Converti la maschera in binaria e in uint8
    mask = (mask > 0).astype(np.uint8)
    
    # Se la maschera ha un canale extra, rimuovilo
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    
    # Converte in immagine binaria (0 e 255)
    mask = mask * 255

    # Assicurati che la maschera abbia le stesse dimensioni (H, W) dell'immagine
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Applica la maschera
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

def histogram_stretching(image):
    min_val = np.min(image)
    max_val = np.max(image)
    stretched = (image - min_val) * (255.0 / (max_val - min_val))
    return stretched.astype(np.uint8)

def gamma_correction(image, gamma=1.2):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def unsharp_mask(image, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return sharpened

def histogram_specification(image, reference):
    image_hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    reference_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])

    cdf_image = image_hist.cumsum()
    cdf_ref = reference_hist.cumsum()

    cdf_image = (cdf_image - cdf_image.min()) * 255 / (cdf_image.max() - cdf_image.min())
    cdf_ref = (cdf_ref - cdf_ref.min()) * 255 / (cdf_ref.max() - cdf_ref.min())

    lut = np.interp(cdf_image, cdf_ref, np.arange(256))
    transformed_image = np.interp(image.flatten(), bins[:-1], lut).reshape(image.shape)
    
    return transformed_image.astype(np.uint8)

def adaptive_gamma_correction(image, alpha=0.5, beta=1.5):
    mean_intensity = np.mean(image) / 255.0  # Normalizza tra 0 e 1
    gamma = beta - (beta - alpha) * mean_intensity  # Adatta gamma dinamicamente
    return gamma_correction(image, gamma)

def resize_image(image, scale_factor=0.5):
    new_rows = int(image.shape[0] * scale_factor)
    new_cols = int(image.shape[1] * scale_factor)

    # Ridimensiona mantenendo il range dei valori originali
    image_resized = transform.resize(image, (new_rows, new_cols), anti_aliasing=True, preserve_range=True)

    return image_resized.astype(image.dtype)  # Mantiene il tipo originale


def enhance_image(image, mask=None, reference=None):
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image

    if mask is not None:
        mask_bin = mask > 0  # Binaria

    # Histogram stretching solo dove necessario
    image_gray = histogram_stretching(image_gray) if mask is None else histogram_stretching(image_gray * mask_bin)

    # Adaptive Gamma Correction
    image_gray = adaptive_gamma_correction(image_gray)

    # Histogram Specification solo se presente un riferimento
    if reference is not None:
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY) if len(reference.shape) == 3 else reference
        image_gray = histogram_specification(image_gray, ref_gray)

    # Applica la maschera per evitare alterazioni fuori dalla regione di interesse
    if mask is not None:
        image_gray = image_gray * mask_bin

    # Applica CLAHE per migliorare il contrasto locale
    image_gray = apply_CLAHE(image_gray)

    # Applica filtro Sobel per enfatizzare i bordi
    image_edges = apply_sobel(image_gray)

    # Combina l'immagine originale con i bordi per enfatizzare il foreground
    enhanced_image = cv2.addWeighted(image_gray, 0.8, image_edges, 0.2, 0)

    return enhanced_image


def generate_superpixels(image, mask, n_superpixels=300):
    """
    Genera superpixel nell'immagine limitata dalla maschera.

    Args:
        image: Immagine in scala di grigi
        mask: Maschera binaria
        n_superpixels: Numero di superpixel da generare

    Returns:
        segments: Mappa di superpixel
    """
    mask_bool = mask > 0  # Converti in booleano
    gray_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    segments = slic(gray_3ch, n_segments=n_superpixels, compactness=2, sigma=2, start_label=1, mask=mask_bool)
    return segments
