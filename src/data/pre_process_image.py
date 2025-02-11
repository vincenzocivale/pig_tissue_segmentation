from PIL import Image
import numpy as np
import cv2

def load_tif_image(image_path):
    with Image.open(image_path) as img:

        # Converti l'immagine in scala di grigi (se è RGB)
        if img.mode != 'L':
            img = img.convert('L') 

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


def enhance_image(img_path, kernel_size=3, iterations=1):
        
        image = load_tif_image(img_path)
        denoised = cv2.GaussianBlur(image, (3,3), 0)
        eroded_image = erode_image(denoised, kernel_size, iterations)
        clahe_image = apply_CLAHE(eroded_image)

        return clahe_image



