from PIL import Image
import numpy as np
import cv2

def load_tif_image(image_path):
    with Image.open(image_path) as img:
        image_array = np.array(img)
    return image_array

def normalize_image(image):
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized_image

def erode_image(image, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=iterations)
    return eroded_image

def apply_otsu_threshold(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    _, otsu_thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_thresholded

def reduce_resolution(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image

def process_image_pipeline(image_path, scale_percent=50, kernel_size=3, iterations=1):
        # Carica l'immagine
        image = load_tif_image(image_path)
        
        # Normalizza l'immagine
        normalized_image = normalize_image(image)
        
        # Riduci la risoluzione dell'immagine
        reduced_image = reduce_resolution(normalized_image, scale_percent)
        
        # Erode l'immagine
        eroded_image = erode_image(reduced_image, kernel_size, iterations)
        
        # Applica la soglia di Otsu
        otsu_thresholded_image = apply_otsu_threshold(eroded_image)
        
        return reduced_image, otsu_thresholded_image

