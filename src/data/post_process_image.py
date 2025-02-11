import cv2
import numpy as np

def extract_external_contours(mask):
    """
    Extracts external contours from a binary mask.

    Parameters:
      - mask: binary grayscale image (0 and 255)

    Returns:
      - contours: list of detected contours
      - mask_contours: image with drawn contours
    """
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_contours = np.zeros_like(mask)

    return contours, mask_contours