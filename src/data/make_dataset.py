import os
from src.data.image_info import ImageSlice

def load_image_slices(folder_root):
    """
    Load ImageSlice objects from a folder and its subfolders.
    It expects the following format for image files:
    - {slice_id}_CH0_CELL_MIP.tif (WGA)
    - {slice_id}_CH1_AUTO_MIP.tif (Autofluorescence)
    - {slice_id}_CH2_COLL_MIP.tif (Collagen)

    Parameters:
    - folder_root: Root directory containing the images.

    Returns:
    - slices: List of ImageSlice objects.
    """
    slices = []
    
    # Walk through the folder_root to find all relevant files
    for root, dirs, files in os.walk(folder_root):
        # Group files by slice_id
        slice_files = {'wga': None, 'collagen': None, 'autofluorescence': None}
        
        # Check all files in the current directory
        for file in files:
            if "CH0_CELL_MIP.tif" in file:
                slice_id = file.split('_')[0]
                slice_files['wga'] = os.path.join(root, file)
            elif "CH1_AUTO_MIP.tif" in file:
                slice_id = file.split('_')[0]
                slice_files['autofluorescence'] = os.path.join(root, file)
            elif "CH2_COLL_MIP.tif" in file:
                slice_id = file.split('_')[0]
                slice_files['collagen'] = os.path.join(root, file)

        # If all 3 images are found, create an ImageSlice object
        if all(slice_files.values()):
            image_slice = ImageSlice(
                slice_id=slice_id,
                path_wga=slice_files['wga'],
                path_collagen=slice_files['collagen'],
                path_autofluorescence=slice_files['autofluorescence']
            )
            slices.append(image_slice)
        else:
            # If one or more images are missing for the slice, raise an error
            missing_files = [key for key, value in slice_files.items() if value is None]
            raise FileNotFoundError(f"Missing files for slice {slice_id}: {', '.join(missing_files)}")

    return slices