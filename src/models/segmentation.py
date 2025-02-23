import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import supervision as sv
from skimage.segmentation import mark_boundaries
from skimage import graph
from skimage.filters import gaussian
from skimage import filters
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

def create_binary_mask_otsu(img, labels):
    """
    Crea una maschera binaria per l'immagine `img` basata sui superpixel.
    La maschera viene creata utilizzando la soglia di Otsu applicata ai colori medi dei superpixel.
    
    Parameters
    ----------
    img : ndarray
        L'immagine originale.
    labels : ndarray
        Le etichette dei superpixel ottenute tramite SLIC o altro metodo di segmentazione.
    
    Returns
    -------
    binary_mask : ndarray
        La maschera binaria risultante dai superpixel.
    """
    # Calcolare il colore medio di ogni superpixel
    mean_colors = np.zeros((np.max(labels), 3))
    for i in range(1, np.max(labels) + 1):
        # Maschera per ogni superpixel
        mask = (labels == i)
        mean_colors[i - 1] = np.mean(img[mask], axis=0)
    
    # Convertire il colore medio in una singola intensità (usando la luminanza)
    # Formula ponderata per la luminanza
    gray_mean_colors = 0.2989 * mean_colors[:, 0] + 0.5870 * mean_colors[:, 1] + 0.1140 * mean_colors[:, 2]
    
    # Applicare la soglia di Otsu
    threshold = filters.threshold_otsu(gray_mean_colors)
    
    # Creare la maschera binaria
    binary_mask = np.zeros_like(labels, dtype=bool)
    
    # Assegnare True ai superpixel che superano la soglia
    for i in range(1, np.max(labels) + 1):
        # Maschera per ogni superpixel
        mask = (labels == i)
        # Se il colore medio di un superpixel è maggiore della soglia, imposta la maschera a True
        if gray_mean_colors[i - 1] >= threshold:
            binary_mask[mask] = True
    
    return binary_mask

def extract_features(image, segments):
    """
    Estrae le feature (intensità, texture, spaziali) dai superpixel senza usare la maschera.

    Args:
        image: Immagine in scala di grigi
        segments: Mappa dei superpixel

    Returns:
        features: Lista di feature per ciascun superpixel
    """
    features = []
    
    for label in np.unique(segments):
        mask_region = (segments == label)

        if np.sum(mask_region) > 0:  # Considera solo i superpixel validi
            intensity = np.mean(image[mask_region])  # Intensità media del superpixel
            texture = cv2.Laplacian(image, cv2.CV_64F)[mask_region].var()  # Variance della texture (Laplaciano)
            spatial = [np.mean(np.where(mask_region)[0] / image.shape[0]), 
                       np.mean(np.where(mask_region)[1] / image.shape[1])]  # Coordinate spaziali normalizzate

            features.append([intensity, texture, *spatial])

    return features


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
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=30, random_state=42)
    kmeans.fit(features_norm)
    
    # Etichette dei cluster
    labels = kmeans.labels_

    return labels

def create_binary_cluster_mask(segments, labels, threshold=0):
    """
    Crea una maschera binaria per l'immagine basata sui cluster.

    Args:
        segments: Mappa dei superpixel
        labels: Etichette dei cluster ottenute da un algoritmo di clustering
        threshold: La soglia per determinare quale cluster è 1 (incluso), tutti gli altri sono 0

    Returns:
        binary_mask: Immagine binaria risultante (1 per un cluster e 0 per gli altri)
    """
    # Creazione della mappa dei cluster (0 o 1 in base alla soglia)
    cluster_map = np.zeros_like(segments)
    
    # Assegna ai superpixel il valore del cluster corrispondente
    for label, cluster in zip(np.unique(segments), labels):
        cluster_map[segments == label] = cluster
    
    # Creare la maschera binaria (con un threshold)
    binary_mask = (cluster_map == threshold).astype(np.uint8)

    return (binary_mask * 255).astype('uint8')


def superpixel_clustering_segmentation(gray,  segments, n_clusters=2):
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


    features = extract_features(gray, segments)
    labels = perform_clustering(features, segments, n_clusters)
    
    mask = create_binary_cluster_mask(segments, labels, 0)

    return mask


def compute_rag(image, segments):
    """Crea un Region Adjacency Graph (RAG) basato sull'intensità media dei superpixel."""
    return graph.rag_mean_color(image, segments, mode='distance')

def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (
        graph.nodes[dst]['total color'] / graph.nodes[dst]['pixel count']
    )

def merge_superpixels(segments, rag, similarity_threshold):
    """Fusione dei superpixel adiacenti con differenze di intensità inferiori alla soglia."""
    return graph.merge_hierarchical(
                                    segments,
                                    rag,
                                    thresh=10,
                                    rag_copy=False,
                                    in_place_merge=True,
                                    merge_func=merge_mean_color,
                                    weight_func=_weight_mean_color,
                                )

def generate_segmented_image(gray, merged_labels, mask, smooth, out_color, transparent):
    """Genera l'immagine segmentata applicando la color map ai cluster ottenuti."""
    if smooth:
        merged_labels = gaussian(merged_labels.astype(np.float32), sigma=2, preserve_range=True)
    
    cmap = plt.get_cmap('viridis', np.max(merged_labels) + 1)
    segmented_viz = np.zeros((*gray.shape, 3), dtype=np.uint8)
    
    for label in np.unique(merged_labels):
        segmented_viz[merged_labels == label] = np.array(cmap(label / np.max(merged_labels))[:3]) * 255
    
    if transparent:
        segmented_viz = np.dstack((segmented_viz, (mask * 255).astype(np.uint8)))
    else:
        segmented_viz[mask == 0] = out_color
    
    return segmented_viz

def superpixel_rag_segmentation(image, mask, segments, similarity_threshold=0.1, 
                                smooth=True, out_color=(0, 0, 0), transparent=False):
    """
    Segmentazione basata su superpixel e Region Adjacency Graph (RAG), limitata alla regione della maschera.
    
    Args:
        image: Immagine in scala di grigi
        mask: Maschera binaria
        segments: Segmenti pre-generati
        similarity_threshold: Soglia di fusione basata sulla differenza di intensità media
        smooth: Se True, applica un filtro per bordi più morbidi
        out_color: Colore RGB per i superpixel fuori dalla maschera (ignorato se transparent=True)
        transparent: Se True, i superpixel fuori dalla maschera saranno trasparenti
    
    Returns:
        (superpixel_viz, segmented_image)
    """
    rag = compute_rag(image, segments)

    merged_labels = merge_superpixels(segments, rag, similarity_threshold)
    segmented_viz = generate_segmented_image(image, merged_labels, mask, smooth, out_color, transparent)
    superpixel_viz = mark_boundaries(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), segments, mode='thick')
    
    return superpixel_viz, segmented_viz
