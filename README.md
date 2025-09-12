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
# --- Erweiterung aufbauend auf deinem ersten Code ---
# Ziel: Metadaten (Zeitstempel) je Bild extrahieren und in einem DataFrame speichern,
#       sodass bei der Ausgabe zunächst nur Pfad + Zeitstempel zu sehen sind.

# Zusätzliche Imports für Metadaten, Zeit und DataFrame
from PIL import ExifTags
from datetime import datetime
import pandas as pd

# Falls du img_paths aus der vorherigen Zelle nicht mehr hast, diese Zeile entkommentieren:
# img_paths = [p for p in IMG_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts]

# Hilfsfunktion: EXIF-String → Python-datetime (EXIF-Zeitstempel haben üblicherweise keine Zeitzone)
def _parse_exif_dt(s: str):
    # Erwartetes EXIF-Format: "YYYY:MM:DD HH:MM:SS"
    try:
        return datetime.strptime(s.strip(), "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None

rows = []

# Hauptschleife über alle gefundenen Bilddateien
for p in img_paths:
    # Standardmäßig leere Werte (None), falls ein Feld fehlt
    dt_original = None     # EXIF: Aufnahmezeit (DateTimeOriginal, Tag 36867)
    dt_digitized = None    # EXIF: Zeitpunkt der Digitalisierung (DateTimeDigitized, Tag 36868)
    dt_generic  = None     # EXIF: Generische Zeit (DateTime, Tag 306)

    # EXIF auslesen (funktioniert typischerweise bei JPEG/TIFF; PNG/BMP haben oft keine EXIF)
    try:
        with Image.open(p) as im:
            exif = im.getexif()
            if exif:
                dto = exif.get(36867)  # DateTimeOriginal
                dtd = exif.get(36868)  # DateTimeDigitized
                dtg = exif.get(306)    # DateTime
                if isinstance(dto, str):
                    dt_original = _parse_exif_dt(dto)
                if isinstance(dtd, str):
                    dt_digitized = _parse_exif_dt(dtd)
                if isinstance(dtg, str):
                    dt_generic = _parse_exif_dt(dtg)
    except Exception:
        # Bild ohne EXIF oder nicht lesbar → Metadaten bleiben None
        pass

    # Dateisystem-Zeiten (unter Windows ist st_ctime ≈ Erstellzeit, st_mtime = letzte Änderung)
    try:
        st = p.stat()
        fs_created = datetime.fromtimestamp(st.st_ctime)
        fs_modified = datetime.fromtimestamp(st.st_mtime)
    except Exception:
        fs_created = None
        fs_modified = None

    # Nur die geforderten Spalten: Pfad + Zeitstempel
    rows.append({
        "Pfad": str(p),                      # Vollständiger Pfad oder Dateiname; hier Pfad als String
        "EXIF_DateTimeOriginal": dt_original,
        "EXIF_DateTimeDigitized": dt_digitized,
        "EXIF_DateTime": dt_generic,
        "FS_Created": fs_created,
        "FS_Modified": fs_modified,
    })

# DataFrame erzeugen und optional nach sinnvoller Zeitpräferenz sortieren
df_meta = pd.DataFrame(rows).sort_values(
    by=["EXIF_DateTimeOriginal", "EXIF_DateTime", "FS_Modified"],
    ascending=True,
    na_position="last"
).reset_index(drop=True)

# Ausgabe des DataFrames (zeigt zunächst nur Pfad und Zeitstempel)
df_meta



# Zelle 2 – Installation & Umgebungscheck (TensorFlow/Keras)

# Installiert in den AKTUELLEN Jupyter-Kernel (wichtig in VS Code).
%pip install --quiet tensorflow tensorflow-hub pillow pandas matplotlib scikit-learn opencv-python

# Minimaler Import- und Versionscheck
import sys, platform
import tensorflow as tf
import tensorflow_hub as hub
import PIL, pandas as pd, matplotlib, sklearn
import cv2

print(f"Python:        {sys.version.split()[0]}  on {platform.system()} {platform.release()}")
print(f"TensorFlow:    {tf.__version__}")
print(f"TF Hub:        {hub.__version__}")
print(f"Pillow:        {PIL.__version__}")
print(f"pandas:        {pd.__version__}")
print(f"matplotlib:    {matplotlib.__version__}")
print(f"scikit-learn:  {sklearn.__version__}")
print(f"OpenCV:        {cv2.__version__}")

# Geräte-Check (GPU/CPU)
gpus = tf.config.list_physical_devices('GPU')
print(f"Gefundene GPUs: {len(gpus)}")
for i, g in enumerate(gpus):
    print(f"  GPU[{i}]: {g.name}")

# Kurzer Sanity-Check einer kleinen TensorFlow-Operation
x = tf.constant([[1.0, 2.0],[3.0, 4.0]])
y = tf.reduce_mean(x)
print("TF Test (Mean):", y.numpy())




# Zelle 3 – Imports, Seeds, Gerätecheck, Pfade, Hyperparameter (TensorFlow/Keras)

from pathlib import Path
import os, sys, platform, random
import numpy as np
import pandas as pd
import tensorflow as tf

# Reproduzierbarkeit: feste Seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    # Ab TF 2.9 verfügbar; macht einige Operationen deterministischer
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

# Geräte- und Umgebungsinfo
gpus = tf.config.list_physical_devices('GPU')
print(f"Python {sys.version.split()[0]} auf {platform.system()} {platform.release()}")
print(f"TensorFlow {tf.__version__} | GPUs erkannt: {len(gpus)}")
for i, g in enumerate(gpus):
    print(f"  GPU[{i}]: {g.name}")

# Pfade: passe diese drei Pfade an deine Struktur an
# IMG_DIR: freier Ordner für ad-hoc Sichtung (hast du in Zelle 4 bereits genutzt)
# DATA_ROOT: Wurzel für Transfer Learning mit splits: data/train/<Klasse>, data/val/<Klasse>, optional data/test/<Klasse>
IMG_DIR   = Path(r"C:\Pfad\zu\deinen\Bildern")
DATA_ROOT = Path(r"C:\Pfad\zu\data")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR   = DATA_ROOT / "val"
TEST_DIR  = DATA_ROOT / "test"   # optional, kann fehlen

# Hyperparameter und Konstanten
IMG_SIZE          = (224, 224)      # für MobileNet/EfficientNet-B0 ausreichend; später ggf. erhöhen
BATCH_SIZE_TRAIN  = 32              # auf CPU ggf. auf 16 reduzieren, wenn RAM knapp ist
BATCH_SIZE_VAL    = 32
EPOCHS_HEAD       = 10              # nur Klassifikationskopf
EPOCHS_FINE       = 10              # selektives Fine-Tuning
LR_HEAD           = 1e-3            # Lernrate für Kopf-Training
LR_FINE           = 1e-5            # kleinere Lernrate fürs Fine-Tuning
LABEL_SMOOTHING   = 0.0             # bei starker Imbalance ggf. 0.05–0.1
AUTOTUNE          = tf.data.AUTOTUNE
NUM_CLASSES       = None            # wird später aus dem Dataset inferiert

# Dateiendungen für die freie Sichtung (Zelle 4) und Metadaten (Zelle 5)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Kurzer Sanity-Print der wichtigsten Einstellungen
print("\nKonfiguration:")
print(f"  IMG_DIR:   {IMG_DIR}")
print(f"  DATA_ROOT: {DATA_ROOT}")
print(f"  TRAIN_DIR: {TRAIN_DIR}")
print(f"  VAL_DIR:   {VAL_DIR}  | TEST_DIR: {TEST_DIR} (optional)")
print(f"  IMG_SIZE:  {IMG_SIZE} | BATCH_TRAIN: {BATCH_SIZE_TRAIN} | BATCH_VAL: {BATCH_SIZE_VAL}")
print(f"  EPOCHS_HEAD: {EPOCHS_HEAD} | EPOCHS_FINE: {EPOCHS_FINE}")
print(f"  LR_HEAD: {LR_HEAD} | LR_FINE: {LR_FINE} | Label Smoothing: {LABEL_SMOOTHING}")


# %% [markdown]
# 🧩 Batch-Video-Cropping direkt per pip (ohne externes ffmpeg)
# Anleitung: Pfade und Parameter unten anpassen und Zelle ausführen.
# Benötigt nur: pip install opencv-python tqdm

# %%
# Abhängigkeiten installieren (funktioniert in Jupyter und VS Code)
import sys, subprocess, pkgutil
def _pip_install(pkg):
    if pkg not in {m.name for m in pkgutil.iter_modules()}:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for p in ["opencv-python", "tqdm"]:
    _pip_install(p)

# %%
import cv2
from pathlib import Path
from tqdm import tqdm

# ====================== EINSTELLUNGEN ======================
INPUT_DIR   = Path("./videos_in")     # Eingangsordner
OUTPUT_DIR  = Path("./videos_out")    # Ausgabeordner
VIDEO_EXTS  = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".webm"}

CROP_MODE   = "percent"               # "percent" oder "px"

# Wenn CROP_MODE == "percent": 0.10 = 10% je Seite
CROP_PCT = dict(left=0.05, right=0.05, top=0.05, bottom=0.05)

# Wenn CROP_MODE == "px": Pixel je Seite
CROP_PX  = dict(left=50, right=50, top=50, bottom=50)

# Optional nur eine Vorschau in Sekunden schreiben (None = ganzes Video)
PREVIEW_SECONDS = None

# Videocodec/FourCC für MP4-Ausgabe (breit kompatibel)
FOURCC = "mp4v"

# Suffix für Ausgabedateinamen
OUTPUT_SUFFIX = "_cropped"
# ===========================================================

def _compute_crop(w, h):
    if CROP_MODE == "percent":
        l = max(0.0, float(CROP_PCT["left"]))
        r = max(0.0, float(CROP_PCT["right"]))
        t = max(0.0, float(CROP_PCT["top"]))
        b = max(0.0, float(CROP_PCT["bottom"]))
        cw = int(round(w * (1.0 - l - r)))
        ch = int(round(h * (1.0 - t - b)))
        cx = int(round(w * l))
        cy = int(round(h * t))
    elif CROP_MODE == "px":
        l = max(0, int(CROP_PX["left"]))
        r = max(0, int(CROP_PX["right"]))
        t = max(0, int(CROP_PX["top"]))
        b = max(0, int(CROP_PX["bottom"]))
        cw = w - l - r
        ch = h - t - b
        cx = l
        cy = t
    else:
        raise ValueError("CROP_MODE muss 'percent' oder 'px' sein.")

    cw = max(2, cw // 2 * 2)  # gerade Dimensionen helfen bei manchen Playern
    ch = max(2, ch // 2 * 2)
    cx = max(0, min(cx, w - 2))
    cy = max(0, min(cy, h - 2))
    if cx + cw > w: cw = max(2, (w - cx) // 2 * 2)
    if cy + ch > h: ch = max(2, (h - cy) // 2 * 2)
    return cw, ch, cx, cy

def _safe_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if fps and fps > 0 else 30.0

def _frame_limit(cap, fps):
    if PREVIEW_SECONDS is None:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    return int(PREVIEW_SECONDS * fps)

def crop_video_file(inp_path: Path, out_path: Path):
    cap = cv2.VideoCapture(str(inp_path))
    if not cap.isOpened():
        raise RuntimeError(f"Konnte {inp_path.name} nicht öffnen.")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = _safe_fps(cap)
    total_frames = _frame_limit(cap, fps)

    cw, ch, cx, cy = _compute_crop(w, h)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (cw, ch))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("VideoWriter konnte nicht geöffnet werden. Prüfe FourCC/Dateiendung.")

    written = 0
    with tqdm(total=total_frames if total_frames is not None else 0, unit="f", disable=total_frames is None) as bar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            crop = frame[cy:cy+ch, cx:cx+cw]
            if crop.shape[1] != cw or crop.shape[0] != ch:
                break
            writer.write(crop)
            written += 1
            if total_frames is not None:
                bar.update(1)
                if written >= total_frames:
                    break

    writer.release()
    cap.release()
    return w, h, cw, ch, cx, cy, written, fps

def process_folder():
    videos = [p for p in INPUT_DIR.glob("*") if p.suffix.lower() in VIDEO_EXTS and p.is_file()]
    if not videos:
        print(f"Keine Videos in {INPUT_DIR.resolve()} gefunden.")
        return
    print(f"{len(videos)} Datei(en) gefunden. Starte Cropping…\n")
    for vid in sorted(videos):
        out_name = f"{vid.stem}{OUTPUT_SUFFIX}.mp4"
        out_path = OUTPUT_DIR / out_name
        try:
            w, h, cw, ch, cx, cy, frames, fps = crop_video_file(vid, out_path)
            print(f"{vid.name}: {w}x{h} → crop {cw}x{ch}+{cx}+{cy}, {frames} Frames @ {fps:.2f} fps → {out_path.name}")
        except Exception as e:
            print(f"Übersprungen ({vid.name}): {e}")
    print("\nFertig. Ausgaben liegen in:", OUTPUT_DIR.resolve())

process_folder()
