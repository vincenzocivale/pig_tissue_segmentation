import cv2
import numpy as np
from src.visualization.visualize import plot_image

import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor




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


def watershed_segmentation(image, exclusion_mask=None):
    """
    Apply watershed segmentation to an image.

    Parameters:
      - image: input image (grayscale or color).
      - exclusion_mask: optional mask to exclude certain regions from segmentation.

    Returns:
      - mask: binary mask obtained by applying watershed segmentation.
    """
    # Check if the image is grayscale
    if len(image.shape) == 2 or image.shape[2] == 1:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Apply exclusion mask if provided
    if exclusion_mask is not None:
        inclusion_mask = cv2.bitwise_not(exclusion_mask)
        gray = cv2.bitwise_and(gray, gray, mask=inclusion_mask)

    # Calculate Otsu's threshold only on the included region
    if exclusion_mask is not None:
        # Extract pixels in the included region
        pixels = gray[inclusion_mask != 0]
        if len(pixels) == 0:
            return np.zeros_like(gray, dtype=np.uint8)
        # Manually calculate Otsu's threshold
        hist, _ = np.histogram(pixels, bins=256, range=(0, 256))
    else:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel()

    total = hist.sum()
    if total == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    
    # Calculate Otsu's threshold manually
    sum_total = np.dot(np.arange(256), hist)
    sum_back = 0
    weight_back = 0
    max_var = 0
    thresh = 0

    for i in range(256):
        weight_back += hist[i]
        if weight_back == 0:
            continue
        weight_fore = total - weight_back
        if weight_fore == 0:
            break
        sum_back += i * hist[i]
        mean_back = sum_back / weight_back
        mean_fore = (sum_total - sum_back) / weight_fore
        var_between = weight_back * weight_fore * (mean_back - mean_fore)**2
        if var_between > max_var:
            max_var = var_between
            thresh = i

    # Apply threshold
    _, thresh = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

    # Dilate the binary mask to ensure well-defined foreground regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_foreground = cv2.dilate(thresh, kernel, iterations=3)

    # Calculate the distance transform
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

    # Threshold the distance transform to identify sure background regions
    _, sure_background = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_background = sure_background.astype(np.uint8)

    unknown = cv2.subtract(sure_foreground, sure_background)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_background)
    markers += 1
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(image_bgr, markers)
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[markers > 1] = 255

    mask = cv2.bitwise_not(mask)
    return mask


def calculate_boundy_box(img):
    
    height, width, _ = img.shape
    bounding_box = np.array([[0, 0, width, height]])

    return bounding_box

def segmentation_with_box(predictor, image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
    predictor.set_image(image)

    bounding_box = calculate_boundy_box(image)

    masks, scores, logits = predictor.predict(
        box=bounding_box,
        multimask_output=False
    )

    if len(masks) == 1:
        return masks[0]
    else:
        raise ValueError("Expected a single mask, but got multiple masks.")
