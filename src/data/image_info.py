class ImageInfo:
    def __init__(self, path, reference_image=None, processed_image=None, segmented_image=None):
        self.path = path
        self.reference_image = reference_image
        self.processed_image = processed_image
        self.segmented_image = segmented_image

    def set_processed_image(self, processed_image):
        self.processed_image = processed_image

    def set_segmented_image(self, segmented_image):
        self.segmented_image = segmented_image

    def get_info(self):
        return {
            "path": self.path,
            "reference_image": self.reference_image,
            "processed_image": self.processed_image,
            "segmented_image": self.segmented_image
        }