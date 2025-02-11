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