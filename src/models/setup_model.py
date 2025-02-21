import subprocess
import os

import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

def setup_sam2(home_directory):

    # Clona il repository SAM2
    subprocess.run(['git', 'clone', 'https://github.com/facebookresearch/segment-anything-2.git'], check=True)
    os.chdir(os.path.join(home_directory, 'segment-anything-2'))

    # Installa le dipendenze
    subprocess.run(['pip', 'install', '-e', '.', '-q'], check=True)

    # Scarica il checkpoint del modello
    checkpoint_url = 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt'
    checkpoint_path = os.path.join(home_directory, 'checkpoints', 'sam2_hiera_large.pt')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    subprocess.run(['wget', '-q', checkpoint_url, '-P', os.path.dirname(checkpoint_path)], check=True)

    
    


def load_sam2(home_directory, device):

    # Configura PyTorch per l'uso della GPU
    if torch.cuda.is_available():
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #checkpoint = os.path.join(home_directory, 'checkpoints', 'sam2_hiera_large.pt')
    checkpoint = r"C:\Users\Admin\Downloads\sam2_hiera_large.pt"
    config = 'sam2_hiera_l.yaml'
    sam2_model = build_sam2(config, checkpoint, device=device, apply_postprocessing=False)

    predictor = SAM2ImagePredictor(sam2_model)

    return predictor
