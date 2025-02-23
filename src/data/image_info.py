import cv2
import os

import numpy as np
import matplotlib.pyplot as plt
import supervision as sv
from skimage.segmentation import mark_boundaries

import src.data.pre_process_image as pi
import src.models.segmentation as seg
import src.data.post_process_image as post
import src.visualization.visualize as vis

class ImageSlice:
    """
    Class to manage a slice consisting of 3 images:
      - wga: WGA 
      - collagen: collagen image
      - autofluorescence: autofluorescence image
    """
    def __init__(self, slice_id, output_folder=r"C:\Users\cical\Documents\GitHub\Repositories\pig_tissue_segmentation\data"):

        self.slice_id = slice_id
        self.output_folder = output_folder

        self.wga = None
        self.collagen = None
        self.autofluorescence = None
        
        self.preprocessed_wga = None
        self.preprocessed_collagen = None
        self.preprocessed_auto = None

        self.superpixel_segments = None

        self.segmented_tissue = None
        self.segmented_cardios = None
        self.segmented_collagen = None

        self.segmented_tissue_post = None
        self.segmented_cardios_post = None

        self.external_contours = None
        self.internal_contours = None

    def load_images(self, path_wga, path_collagen, path_autofluorescence, resize_factor=0.5):
        wga_image = pi.load_tif_image(path_wga)
        collagen_image = pi.load_tif_image(path_collagen)
        autofluorescence_image = pi.load_tif_image(path_autofluorescence)

        self.wga = pi.resize_image(wga_image, resize_factor)
        self.collagen = pi.resize_image(collagen_image, resize_factor)
        self.autofluorescence = pi.resize_image(autofluorescence_image, resize_factor)

    def analyse_image(self, predictor=None):
        slice_folder = os.path.join(self.output_folder, f"slice_{self.slice_id}")
        os.makedirs(slice_folder, exist_ok=True)

        # Use the image of collagen to segment the tissue external contour
        self.preprocessed_collagen = pi.enhance_image(self.collagen)
        self.segmented_tissue = seg.segmentation_with_box(self.preprocessed_collagen, predictor)

        # Apply mask of tissue region to the autofluorescence image and the execute pre processing
        self.preprocessed_auto = pi.enhance_image(self.autofluorescence, self.segmented_tissue)
        self.superpixel_segments = pi.generate_superpixels(self.preprocessed_auto, self.segmented_tissue)

        masks_unsupervised = seg.superpixel_clustering_segmentation(self.preprocessed_auto, mask=self.segmented_tissue , segments=self.superpixel_segments)

        # unsupervised segmentation, so I don't know which is collagen and which is cardios
        self.segmented_cardios = seg.superpixel_clustering_segmentation(self.preprocessed_auto, segments=self.superpixel_segments)
        self.segmented_collagen = self.segmented_cardios
    
    def analyse_image2(self, mask_path):

        # Load the binary mask for tissue segmentation
        self.segmented_tissue = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply mask of tissue region to the autofluorescence image and then execute pre-processing
        self.preprocessed_auto = pi.enhance_image(self.autofluorescence, self.segmented_tissue)
        self.superpixel_segments = pi.generate_superpixels(self.preprocessed_auto, self.segmented_tissue)

        # unsupervised segmentation, so I don't know which is collagen and which is cardios
        self.segmented_cardios = seg.superpixel_clustering_segmentation(self.preprocessed_auto, segments=self.superpixel_segments, tissue_mask=self.segmented_tissue)

        self.segmented_collagen = self.segmented_cardios

        self.segmented_tissue_post, self.external_contours = post.postprocess_mask(self.segmented_tissue)
        self.segmented_cardios_post, self.internal_contours = post.postprocess_mask(self.segmented_cardios)


    def save_results(self):
        if not os.path.isdir(self.output_folder):
            raise  FileNotFoundError(f"Error: The folder to save result '{self.output_folder}' does not exist.")
        
        # saving interim data
        slice_interim_folder = os.path.join(self.output_folder, "interim", f"slice_{self.slice_id}")
        os.makedirs(slice_interim_folder, exist_ok=True)

        cv2.imwrite(os.path.join(slice_interim_folder, f"slice_{self.slice_id}_preprocessed_auto.png"), self.preprocessed_auto)

        superpixel_viz = mark_boundaries(cv2.cvtColor(self.preprocessed_auto, cv2.COLOR_GRAY2RGB), self.superpixel_segments, mode='thick')
        plt.imsave(os.path.join(slice_interim_folder, f"slice_{self.slice_id}_superpixel_viz.png"), superpixel_viz)

        cv2.imwrite(os.path.join(slice_interim_folder, f"slice_{self.slice_id}_segmented_tissue_interim.png"), self.segmented_tissue)
        cv2.imwrite(os.path.join(slice_interim_folder, f"slice_{self.slice_id}_segmented_cardios_interim.png"), self.segmented_cardios)
        cv2.imwrite(os.path.join(slice_interim_folder, f"slice_{self.slice_id}_segmented_collagen_interim.png"), self.segmented_collagen)

         # saving final results
        slice_processed_folder = os.path.join(self.output_folder, "processed", f"slice_{self.slice_id}")
        os.makedirs(slice_processed_folder, exist_ok=True)

        cv2.imwrite(os.path.join(slice_processed_folder, f"slice_{self.slice_id}_segmented_tissue.png"), self.segmented_tissue_post)
        cv2.imwrite(os.path.join(slice_processed_folder, f"slice_{self.slice_id}_segmented_cardios.png"), self.segmented_cardios_post)

        overlayed_image = vis.overlay_contours(self.preprocessed_auto, self.external_contours + self.internal_contours)
        cv2.imwrite(os.path.join(slice_processed_folder, f"slice_{self.slice_id}_segmented_tissue_overlay.png"), overlayed_image)
        

