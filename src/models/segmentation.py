import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import supervision as sv
from skimage.segmentation import mark_boundaries


import src.data.pre_process_image as pi


def calculate_boundy_box(img):
    
    height, width, _ = img.shape
    bounding_box = np.array([[0, 0, width, height]])

    return bounding_box

def segmentation_with_box(image_bgr, predictor) -> np.ndarray:

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


def extract_features(image, segments, mask):
    """
    Estrae le feature (intensità, texture, spaziali) dai superpixel nell'area della maschera.

    Args:
        image: Immagine in scala di grigi
        segments: Mappa dei superpixel
        mask: Maschera binaria

    Returns:
        features: Lista di feature per ciascun superpixel
        valid_labels: Etichette valide dei superpixel
    """
    features = []
    valid_labels = []
    
    for label in np.unique(segments):
        mask_region = (segments == label) & (mask == 1)

        if np.sum(mask_region) > 0:  # Considera solo i superpixel validi
            intensity = np.mean(image[mask_region])
            texture = cv2.Laplacian(image, cv2.CV_64F)[mask_region].var()
            spatial = [np.mean(np.where(mask_region)[0] / image.shape[0]), 
                       np.mean(np.where(mask_region)[1] / image.shape[1])]

            features.append([intensity, texture, *spatial])
            valid_labels.append(label)

    return features, valid_labels

def perform_clustering(features, segments, n_clusters=2):
    """
    Esegue il clustering KMeans sulle feature normalizzate e restituisce una lista di maschere per ciascun cluster.

    Args:
        features: Lista di feature estratte
        segments: Mappa dei superpixel
        n_clusters: Numero di cluster

    Returns:
        masks: Lista di maschere (una per ciascun cluster)
    """
    # Normalizzazione delle feature
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)
    
    # Esecuzione del clustering KMeans
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=50, random_state=42)
    kmeans.fit(features_norm)
    
    # Etichette dei cluster
    labels = kmeans.labels_

    # Creazione della lista di maschere per ciascun cluster
    masks = []
    for cluster in range(n_clusters):
        mask = (segments == cluster).astype(np.uint8)  # Crea maschera per ciascun cluster
        masks.append(mask)

    return masks




def superpixel_clustering_segmentation(gray, mask, segments, n_clusters=2):
    """
    Funzione principale per la segmentazione con superpixel e clustering KMeans.

    Args:
        image_path: Percorso dell'immagine in scala di grigi
        mask_path: Percorso della maschera binaria
        segments: Superpixel in cui è stata suddivisa l'immagine
        n_clusters: Numero di cluster

    Returns:
      
    """
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    else:
        gray = gray

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    else:
        mask = mask

    features, valid_labels = extract_features(gray, segments, mask)
    masks = perform_clustering(features, segments, n_clusters)
    
    return masks