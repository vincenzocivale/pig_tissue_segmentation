import cv2
import os

import src.data.pre_process_image as pi
import src.data.segmentation as seg
import src.data.post_process_image as post

class ImageSlice:
    """
    Class to manage a slice consisting of 3 images:
      - wga: WGA fluorescence image (which can highlight the overall tissue)
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


    def _preprocess(self, ):
        """
        Applies pre-processing (denoising, CLAHE, and erosion) to each image.
        """
        self.preprocessed_wga = pi.enhance_image(self.path_wga)
        self.preprocessed_collagen = pi.enhance_image(self.path_collagen)
        self.preprocessed_auto = pi.enhance_image(self.path_autofluorescence)

    def _segment_image(self):

        self.segmented_tissue = seg.adaptive_thresholding(self.preprocessed_wga)

        # Apply mask of tissue region to the autofluorescence image
        #auto_processed = pi.apply_mask(self.preprocessed_auto, self.segmented_tissue)
        self.segmented_cardios = seg.adaptive_thresholding(auto_processed)
        
        # Remove from tissue mask the regions that are cardios
        self.segmented_collagen = pi.apply_mask(auto_processed, cv2.bitwise_not(self.segmented_cardios))

    def _postprocess(self):
        self.tissue_contours = post.extract_external_contours(self.segmented_tissue)
        self.cardios_contours = post.extract_external_contours(self.segmented_cardios)
        self.collagen_contours = post.extract_external_contours(self.segmented_collagen)

    def _save_results(self):
        if not os.path.isdir(self.output_folder):
            raise  FileNotFoundError(f"Error: The folder to save result '{self.output_folder}' does not exist.")
        
        slice_folder = os.path.join(self.output_folder, f"slice_{self.slice_id}")
        os.makedirs(slice_folder, exist_ok=True)

        cv2.imwrite(os.path.join(slice_folder, f"slice_{self.slice_id}_segmented_tissue.png"), self.segmented_tissue)
        cv2.imwrite(os.path.join(slice_folder, f"slice_{self.slice_id}_segmented_cardios.png"), self.segmented_cardios)
        cv2.imwrite(os.path.join(slice_folder, f"slice_{self.slice_id}_segmented_collagen.png"), self.segmented_collagen)

        # # Create an image with contours
        # wga_image = cv2.imread(self.path_wga, cv2.IMREAD_GRAYSCALE)
        # img_with_contours = cv2.cvtColor(wga_image, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(img_with_contours, self.tissue_contours, -1, (0, 255, 0), 1)  # Green for tissue
        # cv2.drawContours(img_with_contours, self.cardios_contours, -1, (0, 0, 255), 1)  # Red for cardios

        # cv2.imwrite(os.path.join(slice_folder, f"slice_{self.slice_id}_contours.png"), img_with_contours)

    def analyse_image(self):
        self._preprocess()
        self._segment_image()
        self._postprocess()
        self._save_results()