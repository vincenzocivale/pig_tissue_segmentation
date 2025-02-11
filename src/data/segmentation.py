import cv2

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
