import cv2
import numpy as np

def generate_segmented_image(masked_img, kernel):
    edges = cv2.Canny(masked_img, 50, 150)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed_edges