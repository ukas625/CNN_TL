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
# 🔧 Video-Batch-Crop mit FFmpeg (Jupyter-Zelle)
# - Stelle sicher, dass FFmpeg installiert ist (https://ffmpeg.org) und in PATH liegt.
# - Passe die Parameter im Abschnitt "EINSTELLUNGEN" an.
# - Crop wahlweise in Prozent oder Pixeln.
# - Audio/Untertitel werden kopiert, Video wird neu enkodiert (libx264, CRF steuerbar).

# %%
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, Optional

# ========== EINSTELLUNGEN ==========
# Ordner mit Input-Videos und Zielordner
INPUT_DIR = Path(r"./videos_in")
OUTPUT_DIR = Path(r"./videos_out")

# Verarbeite diese Endungen
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".webm"}

# Crop-Modus: "percent" ODER "px"
CROP_MODE = "percent"  # "percent" | "px"

# Wenn CROP_MODE == "percent": Anteile je Seite (0.10 = 10%)
CROP_PCT = {
    "left":   0.05,
    "right":  0.05,
    "top":    0.05,
    "bottom": 0.05,
}

# Wenn CROP_MODE == "px": Pixel je Seite
CROP_PX = {
    "left":   50,
    "right":  50,
    "top":    50,
    "bottom": 50,
}

# Enkodierung: Qualität & Geschwindigkeit (nur Video)
VIDEO_CODEC = "libx264"   # üblich: libx264 oder libx265
CRF = 18                  # 18–23 ist guter Bereich (kleiner = bessere Qualität/größere Datei)
PRESET = "medium"         # ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow

# Optional: nur eine kurze Vorschau jeder Datei rendern (z. B. 3 Sekunden) – gut zum Testen
PREVIEW_SECONDS: Optional[int] = None  # z. B. 3 oder None für komplettes Video

# Suffix für Ausgabedateien
OUTPUT_SUFFIX = "_cropped"
# ===================================


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("FFmpeg/ffprobe nicht gefunden. Bitte installieren und PATH prüfen.")


def ffprobe_size(path: Path) -> Tuple[int, int]:
    """Ermittelt die sichtbare Breite/Höhe (Rotation berücksichtigt)."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", str(path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe-Fehler bei {path}: {res.stderr}")
    data = json.loads(res.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"Keine Videostreams in {path}")
    s = streams[0]
    w, h = int(s["width"]), int(s["height"])

    # Rotation berücksichtigen (falls als Tag vorhanden)
    rotate = 0
    tags = s.get("tags", {}) or {}
    if "rotate" in tags:
        try:
            rotate = int(tags["rotate"]) % 180
        except Exception:
            rotate = 0

    # Manche Container speichern Rotation in side_data_list
    sdl = s.get("side_data_list", []) or []
    for item in sdl:
        if item.get("rotation"):
            try:
                if int(item["rotation"]) % 180:
                    rotate = 90
            except Exception:
                pass

    if rotate == 90:
        w, h = h, w
    return w, h


def compute_crop(w: int, h: int) -> Tuple[int, int, int, int]:
    """Berechnet crop=w:h:x:y basierend auf Einstellungen."""
    if CROP_MODE == "percent":
        l = max(0.0, CROP_PCT["left"])
        r = max(0.0, CROP_PCT["right"])
        t = max(0.0, CROP_PCT["top"])
        b = max(0.0, CROP_PCT["bottom"])
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

    # Sicherstellen, dass Werte sinnvoll sind und durch 2 teilbar (für einige Codecs hilfreich)
    cw = max(2, cw // 2 * 2)
    ch = max(2, ch // 2 * 2)
    cx = max(0, min(cx, w - 2))
    cy = max(0, min(cy, h - 2))

    # Wenn die Box außerhalb rutscht, clampen
    if cx + cw > w:
        cw = max(2, (w - cx) // 2 * 2)
    if cy + ch > h:
        ch = max(2, (h - cy) // 2 * 2)

    return cw, ch, cx, cy


def build_ffmpeg_cmd(inp: Path, out: Path, crop: Tuple[int, int, int, int]) -> list:
    cw, ch, cx, cy = crop
    vf = f"crop={cw}:{ch}:{cx}:{cy}"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]

    if PREVIEW_SECONDS:
        # Kurzes Snippet ab Sekunde 1; passt die Startzeit bei Bedarf an
        cmd += ["-ss", "1", "-t", str(PREVIEW_SECONDS)]

    cmd += ["-i", str(inp), "-filter:v", vf,
            "-c:v", VIDEO_CODEC, "-crf", str(CRF), "-preset", PRESET,
            "-c:a", "copy", "-c:s", "copy", "-movflags", "+faststart",
            str(out)]
    return cmd


def process_all():
    ensure_ffmpeg()
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS and p.is_file()]
    if not videos:
        print(f"Keine Videos in {INPUT_DIR.resolve()} gefunden. Unterstützte Endungen: {sorted(VIDEO_EXTS)}")
        return

    print(f"{len(videos)} Datei(en) gefunden. Starte Crop…\n")

    for i, vid in enumerate(sorted(videos), 1):
        try:
            w, h = ffprobe_size(vid)
            cw, ch, cx, cy = compute_crop(w, h)

            # Ausgabe als MP4; Dateinamen mit Suffix
            out_name = f"{vid.stem}{OUTPUT_SUFFIX}.mp4"
            out_path = OUTPUT_DIR / out_name

            cmd = build_ffmpeg_cmd(vid, out_path, (cw, ch, cx, cy))
            print(f"[{i}/{len(videos)}] {vid.name}: {w}x{h} -> crop {cw}x{ch}+{cx}+{cy}")
            subprocess.run(cmd, check=True)

        except subprocess.CalledProcessError as e:
            print(f"FFmpeg-Fehler bei {vid.name}: {e}")
        except Exception as e:
            print(f"Übersprungen ({vid.name}): {e}")

    print("\nFertig. Ausgaben liegen in:", OUTPUT_DIR.resolve())


process_all()
