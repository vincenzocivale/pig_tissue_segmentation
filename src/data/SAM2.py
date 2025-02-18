import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Funzione per caricare l'immagine
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return np.array(image)

# Funzione per gestire i clic del mouse
def onclick(event, predictor, image):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        print(f"Coordinate selezionate: ({x}, {y})")
        
        # Creazione del prompt con il punto selezionato
        input_prompts = {
            "point_coords": torch.tensor([[x, y]], device=predictor.device),
            "point_labels": torch.tensor([1], device=predictor.device)  # 1 per punto positivo
        }
        
        # Esecuzione della predizione
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, _, _ = predictor.predict(input_prompts)
        
        # Visualizzazione del risultato
        mask = masks[0].cpu().numpy()
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5, cmap='jet')
        plt.title("Segmentazione con SAM2")
        plt.axis('off')
        plt.show()

# Percorso dell'immagine di input
image_path = r"C:\Users\cical\Documents\GitHub\Repositories\pig_tissue_segmentation\data\processed\slice_BZ6\slice_BZ6_preprocessed_collagen.png"

# Caricamento dell'immagine
image = load_image(image_path)

# Inizializzazione del predittore SAM2
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
predictor.set_image(image)

# Visualizzazione dell'immagine e attesa del clic dell'utente
fig, ax = plt.subplots()
ax.imshow(image)
ax.set_title("Clicca per selezionare un punto di interesse")
cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, predictor, image))
plt.axis('off')
plt.show()
