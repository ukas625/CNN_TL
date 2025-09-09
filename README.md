# CNN_TL

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Pfad zu deinem Bildordner anpassen (z.B. r"C:\Users\deinname\Bilder\goPro" oder "/Users/deinname/Bilder/goPro")
IMG_DIR = Path(r"/Pfad/zu/deinen/Bildern")

# Ein paar Bilddateien einsammeln
exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
img_paths = [p for p in IMG_DIR.iterdir() if p.suffix.lower() in exts]

print(f"Gefundene Bilder: {len(img_paths)}")
for p in img_paths[:5]:  # zeige die ersten 5
    img = Image.open(p).convert("RGB")
    plt.figure()
    plt.imshow(np.asarray(img))
    plt.title(p.name)
    plt.axis("off")
plt.show()
