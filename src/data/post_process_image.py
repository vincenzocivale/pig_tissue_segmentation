import cv2
import numpy as np
from scipy.interpolate import splprep, splev

def clean_mask(mask, kernel_size=5):
    """Rimuove rumore e artefatti dalla maschera binaria usando operazioni morfologiche."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Rimuove piccoli pixel isolati
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)  # Riempie buchi nella maschera
    return mask_cleaned

def smooth_mask(mask, blur_size=5):
    """Applica un filtro Gaussiano per lisciare i bordi della maschera."""
    return cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

def extract_contours(mask, min_area=500):
    """Trova tutti i contorni nella maschera e filtra quelli troppo piccoli."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

def create_final_mask(shape, contours):
    """Genera una nuova maschera binaria con tutti i contorni processati."""
    final_mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(final_mask, contours, -1, 255, thickness=cv2.FILLED)
    return final_mask

def postprocess_mask(mask):
    """Pipeline completa per il post-processing della maschera mantenendo tutti i contorni."""
    mask = clean_mask(mask)  # Pulizia artefatti
    mask = smooth_mask(mask)  # Lisciatura
    contours = extract_contours(mask)  # Estrazione di tutti i contorni significativi
    final_mask = create_final_mask(mask.shape, contours)  # Nuova maschera con tutti i contorni
    
    return final_mask, contours