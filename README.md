from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

IMG_DIR = Path(r"C:\Pfad\zu\deinen\Bildern")  # unter Windows Raw-String nehmen
exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

img_paths = [p for p in IMG_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts]
print(f"Gefundene Bilder: {len(img_paths)}")

for p in img_paths[:5]:
    img = Image.open(p).convert("RGB")
    plt.figure()
    plt.imshow(np.asarray(img))
    plt.title(p.name)
    plt.axis("off")
plt.show()
