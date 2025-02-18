import cv2
import numpy as np
from src.data.image_info import ImageInfo

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
