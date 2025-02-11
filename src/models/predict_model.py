import cv2
import numpy as np
from src.data.image_info import ImageInfo

def generate_segmented_image(img: ImageInfo, kernel):
    edges = cv2.Canny(img.processed_image, 50, 150)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    img.set_segmented_image(closed_edges)