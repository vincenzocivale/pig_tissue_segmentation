from PIL import Image
import numpy as np
import cv2

import os
from shutil import copyfile


class ImageInfo:
    def __init__(self, path, fluorescence_image=None, collagen_image=None, cellular_image=None, segmented_overall=None, segmented_cellular=None, segmented_collagen=None):
        self.path = path
        self.fluorescence_image = fluorescence_image
        self.collagen_image = collagen_image
        self.cellular_image = cellular_image
        self.segmented_overall = segmented_overall
        self.segmented_cellular = segmented_cellular
        self.segmented_collagen = segmented_collagen
    
    def set_fluorescence_image(self, fluorescence_image):
        self.fluorescence_image = fluorescence_image

    def set_collagen_image(self, collagen_image):
        self.collagen_image = collagen_image

    def set_cellular_image(self, cellular_image):
        self.cellular_image = cellular_image

    def set_segmented_overall(self, segmented_overall):
        self.segmented_overall = segmented_overall

    def set_segmented_cellular(self, segmented_cellular):
        self.segmented_cellular = segmented_cellular

    def set_segmented_collagen(self, segmented_collagen):
        self.segmented_collagen = segmented_collagen

    def get_info(self):
        return {
            "path": self.path,
            "fluorescence_image": self.fluorescence_image,
            "collagen_image": self.collagen_image,
            "cellular_image": self.cellular_image,
            "segmented_overall": self.segmented_overall,
            "segmented_cellular": self.segmented_cellular,
            "segmented_collagen": self.segmented_collagen
        }
    
    def save_info(self):
        interim_data_folder_root = "data\\interim"
        # Create the directory if it doesn't exist
        if not os.path.exists(interim_data_folder_root):
            os.makedirs(interim_data_folder_root)

        # Save each image in the appropriate subfolder
        if self.fluorescence_image:
            fluorescence_folder = os.path.join(interim_data_folder_root, "fluorescence")
            if not os.path.exists(fluorescence_folder):
                os.makedirs(fluorescence_folder)
            fluorescence_path = os.path.join(fluorescence_folder, os.path.basename(self.path))
            copyfile(self.fluorescence_image, fluorescence_path)

        if self.collagen_image:
            collagen_folder = os.path.join(interim_data_folder_root, "collagen")
            if not os.path.exists(collagen_folder):
                os.makedirs(collagen_folder)
            collagen_path = os.path.join(collagen_folder, os.path.basename(self.path))
            copyfile(self.collagen_image, collagen_path)

        if self.cellular_image:
            cellular_folder = os.path.join(interim_data_folder_root, "cellular")
            if not os.path.exists(cellular_folder):
                os.makedirs(cellular_folder)
            cellular_path = os.path.join(cellular_folder, os.path.basename(self.path))
            copyfile(self.cellular_image, cellular_path)

        if self.segmented_overall:
            segmented_overall_folder = os.path.join(interim_data_folder_root, "segmented_overall")
            if not os.path.exists(segmented_overall_folder):
                os.makedirs(segmented_overall_folder)
            segmented_overall_path = os.path.join(segmented_overall_folder, os.path.basename(self.path))
            copyfile(self.segmented_overall, segmented_overall_path)

        if self.segmented_cellular:
            segmented_cellular_folder = os.path.join(interim_data_folder_root, "segmented_cellular")
            if not os.path.exists(segmented_cellular_folder):
                os.makedirs(segmented_cellular_folder)
            segmented_cellular_path = os.path.join(segmented_cellular_folder, os.path.basename(self.path))
            copyfile(self.segmented_cellular, segmented_cellular_path)

        if self.segmented_collagen:
            segmented_collagen_folder = os.path.join(interim_data_folder_root, "segmented_collagen")
            if not os.path.exists(segmented_collagen_folder):
                os.makedirs(segmented_collagen_folder)
            segmented_collagen_path = os.path.join(segmented_collagen_folder, os.path.basename(self.path))
            copyfile(self.segmented_collagen, segmented_collagen_path)

def load_tif_image(image_path):
    with Image.open(image_path) as img:
        image_array = np.array(img)
    return image_array

def apply_CLAHE(image, clipLimit=2.0, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahe_image = clahe.apply(image)

    return clahe_image

def normalize_image(image):
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized_image


def apply_top_hat(image, kernel_size=15):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

def erode_image(image, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)

def apply_otsu_threshold(image):
    _, otsu_thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_thresholded

def reduce_resolution(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def process_image_pipeline(img: ImageInfo, scale_percent=50, kernel_size=3, iterations=1):
        # Carica l'immagine
        image = load_tif_image(img.path)
        
        # Riduci la risoluzione
        reduced_image = reduce_resolution(image, scale_percent)

        # Applica CLAHE per migliorare il contrasto
        nromalized_image = normalize_image(reduced_image)

        # Applica filtro morfologico Top-Hat per separare il foreground
        #top_hat_image = apply_top_hat(nromalized_image)

        # Erode per rimuovere artefatti
        eroded_image = erode_image(nromalized_image, kernel_size, iterations)

        # Applica la soglia di Otsu per segmentare
        #img.set_processed_image(apply_otsu_threshold(eroded_image))

        cv2.imshow("Processed Image", apply_otsu_threshold(eroded_image))
        cv2.waitKey(0)
        

img = ImageInfo(r"C:\Users\cical\Documents\GitHub\Repositories\pig_tissue_segmentation\data\raw\BZ1_CH0_CELL_MIP.tif")
process_image_pipeline(img)


