import cv2
import numpy as np
from sklearn.cluster import KMeans

import sys
project_home_dir= r"D:\Repositories\pig_tissue_segmentation-main"
sys.path.append(project_home_dir)

import supervision as sv




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

def segmentation_with_box(predictor, image_path) -> np.ndarray:
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
    predictor.set_image(image_rgb)

    bounding_box = calculate_boundy_box(image_rgb)

    masks, scores, logits = predictor.predict(
        box=bounding_box,
        multimask_output=False
    )

    if len(masks) == 1:
       #return annotate_mask_only(image, masks)
       return np.logical_not(masks.astype(bool))
    else:
        raise ValueError("Expected a single mask, but got multiple masks.")
    

def annotate_mask_only(image, masks):
        #masks = np.logical_not(masks.astype(bool))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Creiamo l'annotatore per la maschera
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

        # Creiamo l'oggetto `Detections` con solo la maschera
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks.astype(bool)
        )

        segmented_image = mask_annotator.annotate(scene=image.copy(), detections=detections)

        return segmented_image

def double_otsu_thresholding(image):
    """
    Apply double Otsu's thresholding to segment the image into
    background, foreground, and uncertain regions.

    Parameters:
        image (numpy.ndarray): Grayscale image.

    Returns:
        tuple: Binary masks for background, foreground, and uncertain regions.
    """
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = image
    
    # First Otsu's threshold
    _, threshold1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert the image for the second Otsu's threshold
    _, threshold2 = cv2.threshold(255 - image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Create masks for background, foreground, and uncertain regions
    background_mask = (image <= threshold1).astype(np.uint8)
    foreground_mask = (image > 255 - threshold2).astype(np.uint8)
    uncertain_mask = ((image > threshold1) & (image <= 255 - threshold2)).astype(np.uint8)
    
    return background_mask, foreground_mask, uncertain_mask


def apply_otsu_threshold(image_path, mask=None):
    """
    Applica la soglia di Otsu su un'immagine utilizzando una maschera opzionale per definire la ROI.

    Args:
        image_path (str): Percorso dell'immagine di input.
        mask_path (str, opzionale): Percorso della maschera. Se None, l'intera immagine viene utilizzata come ROI.

    Returns:
        binary_mask (numpy.ndarray): Maschera binaria risultante dopo l'applicazione della soglia di Otsu.
    """
    # Carica l'immagine in scala di grigi
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Impossibile caricare l'immagine dal percorso: {image_path}")
    
    if mask is not None:
        # Assicurati che la maschera sia binaria
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Applica la maschera all'immagine per ottenere la ROI
        roi = cv2.bitwise_and(image, image, mask=mask)

        # Calcola l'istogramma dei pixel nella ROI
        hist = cv2.calcHist([roi], [0], mask, [256], [0, 256])

        # Calcola la soglia di Otsu utilizzando l'istogramma
        total = np.sum(hist)
        current_max, threshold = 0, 0
        sum_total, sum_foreground = 0, 0
        weight_background, weight_foreground = 0, 0

        for i in range(256):
            sum_total += i * hist[i]
            weight_background += hist[i]
            if weight_background == 0:
                continue
            weight_foreground = total - weight_background
            if weight_foreground == 0:
                break
            sum_foreground += i * hist[i]
            mean_background = sum_total / weight_background
            mean_foreground = (sum_total - sum_foreground) / weight_foreground
            between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
            if between_class_variance > current_max:
                current_max = between_class_variance
                threshold = i

        # Applica la soglia calcolata all'intera immagine
        _, binary_mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Applica la maschera all'immagine per ottenere la ROI
        roi = cv2.bitwise_and(image, image, mask=mask)

        # Calcola l'istogramma dei pixel nella ROI
        hist = cv2.calcHist([roi], [0], mask, [256], [0, 256])

        # Calcola la soglia di Otsu utilizzando l'istogramma
        total = np.sum(hist)
        current_max, threshold = 0, 0
        sum_total, sum_foreground = 0, 0
        weight_background, weight_foreground = 0, 0

        for i in range(256):
                sum_total += i * hist[i]
                weight_background += hist[i]
                if weight_background == 0:
                    continue
                weight_foreground = total - weight_background
                if weight_foreground == 0:
                    break
                sum_foreground += i * hist[i]
                mean_background = sum_total / weight_background
                mean_foreground = (sum_total - sum_foreground) / weight_foreground
                between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
                if between_class_variance > current_max:
                    current_max = between_class_variance
                    threshold = i

        # Applica la soglia calcolata all'intera immagine
        _, binary_mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    else:
        # Se non viene fornita una maschera, applica direttamente la soglia di Otsu all'intera immagine
        _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_mask


def select_foreground_seeds(foreground_mask, num_seeds=1):
    """
    Select seed points from the foreground mask.

    Parameters:
        foreground_mask (numpy.ndarray): Binary mask of the foreground region.
        num_seeds (int): Number of seed points to select.

    Returns:
        list: List of [x, y] coordinates for the selected seed points.
    """
    # Find contours in the foreground mask
    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    seeds = []
    for contour in contours:
        # Calculate the centroid of each contour
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            seeds.append([cx, cy])
    
    # Sort seed points based on their distance from the image center
    height, width = foreground_mask.shape
    center = np.array([width // 2, height // 2])
    seeds.sort(key=lambda s: np.linalg.norm(np.array(s) - center))
    
    # Return the specified number of seed points
    return np.array(seeds[:num_seeds])

def segmentation_with_seeds(predictor, image) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    predictor.set_image(image)

    background_mask, foreground_mask, uncertain_mask = double_otsu_thresholding(image)

    # Select seed points for the foreground
    foreground_seeds = select_foreground_seeds(foreground_mask, num_seeds=10)

    masks, scores, logits = predictor.predict(
    point_coords=foreground_seeds,
    point_labels= np.ones(foreground_seeds.shape[0]),
    multimask_output=False,
    )

    return np.logical_not(masks.astype(bool))


def cluster_image(image_path, num_clusters, use_spatial=False, spatial_weight=0.1):
    """
    Segmenta un'immagine in 'num_clusters' cluster utilizzando k-means.
    
    Parametri:
    - image_path: percorso dell'immagine da segmentare.
    - num_clusters: numero di cluster in cui dividere l'immagine.
    - use_spatial: se True, le coordinate spaziali vengono aggiunte come feature.
    - spatial_weight: peso relativo delle coordinate spaziali (utile per bilanciare il contributo del colore).
    
    Ritorna:
    - segmented_image: immagine segmentata in cui ogni pixel assume il colore del suo cluster.
    """
    # Carica l'immagine e convertila in RGB
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image_rgb.shape

    # Prepara le feature per il clustering
    if not use_spatial:
        # Usa solo le informazioni di colore: reshape in (N, 3)
        features = image_rgb.reshape((-1, 3))
        features = np.float32(features)
    else:
        # Crea una matrice che includa anche le coordinate spaziali
        # Normalizza i valori di colore nell'intervallo [0, 1]
        color_features = image_rgb.reshape((-1, 3)).astype(np.float32) / 255.0
        # Crea una meshgrid delle coordinate (x, y)
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        X = X.reshape((-1, 1)).astype(np.float32) / w  # normalizzazione
        Y = Y.reshape((-1, 1)).astype(np.float32) / h  # normalizzazione
        # Concatena le feature di colore con quelle spaziali (eventualmente pesate)
        features = np.concatenate((color_features, spatial_weight * X, spatial_weight * Y), axis=1)

    # Applica il clustering k-means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(features)

    # Estrai i centroidi e ricostruisci l'immagine segmentata
    if not use_spatial:
        centers = np.uint8(kmeans.cluster_centers_)
    else:
        # Se abbiamo aggiunto le coordinate, consideriamo solo le prime 3 colonne per il colore
        centers = (kmeans.cluster_centers_[:, :3] * 255).astype(np.uint8)

    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(image_rgb.shape)

    return segmented_image