import cv2
import os

import numpy as np
import matplotlib.pyplot as plt
import supervision as sv

import src.data.pre_process_image as pi
import src.models.segmentation as seg
import src.data.post_process_image as post

class ImageSlice:
    """
    Class to manage a slice consisting of 3 images:
      - wga: WGA 
      - collagen: collagen image
      - autofluorescence: autofluorescence image
    """
    def __init__(self, slice_id, path_wga, path_collagen, path_autofluorescence, output_folder=r"C:\Users\cical\Documents\GitHub\Repositories\pig_tissue_segmentation\data\processed"):
        self.slice_id = slice_id
        self.path_wga = path_wga
        self.path_collagen = path_collagen
        self.path_autofluorescence = path_autofluorescence
        self.output_folder = output_folder
        
        self.preprocessed_wga = None
        self.preprocessed_collagen = None
        self.preprocessed_auto = None

        self.segmented_tissue = None
        self.segmented_cardios = None
        self.segmented_collagen = None

        self.tissue_contours = None
        self.cardios_contours = None
        self.collagen_contours = None

    def analyse_image(self, predictor=None):
        slice_folder = os.path.join(self.output_folder, f"slice_{self.slice_id}")
        os.makedirs(slice_folder, exist_ok=True)

        # Use the image of collagen to segment the tissue external contour
        self.preprocessed_collagen = pi.enhance_image(self.path_collagen)
        processed_collagen_path = os.path.join(slice_folder, f"slice_{self.slice_id}_preprocessed_collagen.png")
        cv2.imwrite(processed_collagen_path, self.preprocessed_collagen)

        #tissue_mask = seg.segmentation_with_box(predictor,processed_collagen_path)
        tissue_mask = seg.cluster_image(processed_collagen_path)
        #self.segmented_tissue= seg.annotate_mask_only(self.preprocessed_collagen, tissue_mask)
        self.segmented_tissue = tissue_mask

        # Apply mask of tissue region to the autofluorescence image and the execute pre processing
        self.preprocessed_auto = pi.enhance_image(self.path_autofluorescence, tissue_mask)
        preprocessed_auto_path = os.path.join(slice_folder, f"slice_{self.slice_id}_preprocessed_auto.png")
        cv2.imwrite(preprocessed_auto_path, self.preprocessed_auto)

        #cardio_mask = seg.segmentation_with_box(predictor, preprocessed_auto_path)
        cardio_mask = seg.cluster_image(preprocessed_auto_path, tissue_mask)
        #self.segmented_cardios= seg.annotate_mask_only(self.preprocessed_collagen, cardio_mask)
        self.segmented_cardios = cardio_mask

        cv2.imwrite(os.path.join(slice_folder, f"slice_{self.slice_id}_segmented_tissue.png"), self.segmented_tissue)
        cv2.imwrite(os.path.join(slice_folder, f"slice_{self.slice_id}_segmented_cardios.png"), self.segmented_cardios)
       
        # DA MODIFICARE
        self.segmented_collagen = self.segmented_cardios

        #self._save_results()



    def _postprocess(self):
        self.tissue_contours = post.extract_smoothed_contours(self.segmented_tissue)
        self.cardios_contours = post.extract_smoothed_contours(self.segmented_cardios)
        self.collagen_contours = post.extract_smoothed_contours(self.segmented_collagen)

    def _save_results(self):
        if not os.path.isdir(self.output_folder):
            raise  FileNotFoundError(f"Error: The folder to save result '{self.output_folder}' does not exist.")
        
        slice_folder = os.path.join(self.output_folder, f"slice_{self.slice_id}")
        os.makedirs(slice_folder, exist_ok=True)

        # cv2.imwrite(os.path.join(slice_folder, f"slice_{self.slice_id}_preprocessed_wga.png"), self.preprocessed_wga)
        cv2.imwrite(os.path.join(slice_folder, f"slice_{self.slice_id}_preprocessed_collagen.png"), self.preprocessed_collagen)
        cv2.imwrite(os.path.join(slice_folder, f"slice_{self.slice_id}_preprocessed_auto.png"), self.preprocessed_auto)

        cv2.imwrite(os.path.join(slice_folder, f"slice_{self.slice_id}_segmented_tissue.png"), self.segmented_tissue)
        cv2.imwrite(os.path.join(slice_folder, f"slice_{self.slice_id}_segmented_cardios.png"), self.segmented_cardios)
        cv2.imwrite(os.path.join(slice_folder, f"slice_{self.slice_id}_segmented_collagen.png"), self.segmented_collagen)

        # # Create an image with contours
        # contours_tissue = post.draw_dashed_contours(self.preprocessed_auto, self.tissue_contours, color=(0, 255, 0))
        # contours_cardios = post.draw_dashed_contours(self.preprocessed_auto, self.cardios_contours, color=(0, 0, 255))
        # cv2.imwrite(os.path.join(slice_folder, f"slice_{self.slice_id}_contours_tissue.png"), contours_tissue)
        # cv2.imwrite(os.path.join(slice_folder, f"slice_{self.slice_id}_contours_cardios.png"), contours_cardios)
    
    # def analyse_image(self):
    #     self._preprocess()
    #     self._segment_image()
    #     self._postprocess()
    #     self._save_results()