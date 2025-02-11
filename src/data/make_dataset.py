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
    slices = dict()
    
    # Walk through the folder_root to find all relevant files
    for root, dirs, files in os.walk(folder_root):
        # Group files by slice_id
        slice_files = {'wga': None, 'collagen': None, 'autofluorescence': None}
        
        # Check all files in the current directory
        for file in files:
            if "CH0_CELL_MIP.tif" in file:
                slice_id = file.split('_')[0]
                if slice_id not in slices.keys():
                    slices[slice_id] ={'wga': None, 'collagen': None, 'autofluorescence': None}
                slices[slice_id]['wga'] = os.path.join(root, file)
            elif "CH1_AUTO_MIP.tif" in file:
                slice_id = file.split('_')[0]
                if slice_id not in slices.keys():
                    slices[slice_id] ={'wga': None, 'collagen': None, 'autofluorescence': None}
                slices[slice_id]['autofluorescence'] = os.path.join(root, file)
            elif "CH2_COLL_MIP.tif" in file:
                slice_id = file.split('_')[0]
                if slice_id not in slices.keys():
                    slices[slice_id] ={'wga': None, 'collagen': None, 'autofluorescence': None}
                slices[slice_id]['collagen'] = os.path.join(root, file)

    return slices
