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
# 🧩 Lokales Video-Cropping mit minimalem Qualitätsverlust (nur pip: av, numpy, tqdm)
# - Qualitätsmodi: "lossless_ffv1" (MKV, wirklich verlustfrei), "lossless_h264" (CRF 0), "visually_lossless_h264"
# - Audio standardmäßig verlustfrei als PCM (groß, aber sicher). Optional AAC aktivierbar.
# - Alles lokal, keine externen Tools nötig.

# %%
import sys, subprocess, pkgutil
def _pip_install(pkg):
    if pkg not in {m.name for m in pkgutil.iter_modules()}:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for p in ["av", "numpy", "tqdm"]:
    _pip_install(p)

# %%
from pathlib import Path
from typing import Optional, Tuple
import av
import numpy as np
from tqdm import tqdm

# ====================== EINSTELLUNGEN ======================
INPUT_DIR   = Path("./videos_in")
OUTPUT_DIR  = Path("./videos_out")
VIDEO_EXTS  = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".webm"}

CROP_MODE   = "percent"               # "percent" oder "px"
CROP_PCT    = dict(left=0.05, right=0.05, top=0.05, bottom=0.05)   # bei percent: Anteile je Seite
CROP_PX     = dict(left=50, right=50, top=50, bottom=50)           # bei px: Pixel je Seite

QUALITY_MODE = "lossless_ffv1"        # "lossless_ffv1" | "lossless_h264" | "visually_lossless_h264"
VISUAL_CRF   = 10                     # nur für "visually_lossless_h264" (kleiner = bessere Qualität)
H264_PRESET  = "slow"                 # "ultrafast" ... "veryslow"

AUDIO_MODE   = "pcm_s16le"            # "pcm_s16le" (verlustfrei, groß) | "aac"
AUDIO_BITRATE = "192k"                # nur falls AAC genutzt wird

PREVIEW_SECONDS: Optional[int] = None # z. B. 5 oder None für komplettes Video
OUTPUT_SUFFIX = "_cropped"
# ===========================================================

def _compute_crop(w: int, h: int) -> Tuple[int,int,int,int]:
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
        cx, cy = l, t
    else:
        raise ValueError("CROP_MODE muss 'percent' oder 'px' sein.")
    cw = max(2, cw // 2 * 2)
    ch = max(2, ch // 2 * 2)
    cx = max(0, min(cx, max(0, w-2)))
    cy = max(0, min(cy, max(0, h-2)))
    if cx + cw > w: cw = max(2, (w - cx) // 2 * 2)
    if cy + ch > h: ch = max(2, (h - cy) // 2 * 2)
    return cw, ch, cx, cy

def _out_params(mode: str):
    if mode == "lossless_ffv1":
        container = "mkv"
        vcodec = "ffv1"
        vopts  = {"level":"3", "coder":"1", "context":"1", "g":"1"}  # intra, verlustfrei
        pixfmt = "rgb24"  # wirklich ohne Chroma-Subsampling
    elif mode == "lossless_h264":
        container = "mp4"
        vcodec = "libx264"
        vopts  = {"crf":"0", "preset":H264_PRESET, "tune":"grain"}
        pixfmt = "yuv444p"  # 4:4:4 für verlustfreie Pfade
    elif mode == "visually_lossless_h264":
        container = "mp4"
        vcodec = "libx264"
        vopts  = {"crf":str(VISUAL_CRF), "preset":H264_PRESET, "tune":"film"}
        pixfmt = "yuv420p"  # maximale Kompatibilität
    else:
        raise ValueError("Ungültiger QUALITY_MODE.")
    return container, vcodec, vopts, pixfmt

def _audio_params():
    if AUDIO_MODE == "pcm_s16le":
        return "pcm_s16le", {"ar":"48000", "ac":"2"}     # 48 kHz, Stereo
    elif AUDIO_MODE == "aac":
        return "aac", {"bit_rate":AUDIO_BITRATE}
    else:
        raise ValueError("Ungültiger AUDIO_MODE.")

def crop_video(inp_path: Path):
    with av.open(str(inp_path), mode="r") as in_ctr:
        v_in = next((s for s in in_ctr.streams if s.type=="video"), None)
        a_in = next((s for s in in_ctr.streams if s.type=="audio"), None)
        if v_in is None:
            raise RuntimeError("Kein Videostream gefunden.")

        # Zielparameter bestimmen
        container_ext, vcodec, vopts, pixfmt = _out_params(QUALITY_MODE)
        out_name = f"{inp_path.stem}{OUTPUT_SUFFIX}.{container_ext}"
        out_path = OUTPUT_DIR / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Abspielrate und Crop berechnen
        rate = float(v_in.average_rate) if v_in.average_rate else 30.0
        w_in, h_in = v_in.codec_context.width, v_in.codec_context.height
        cw, ch, cx, cy = _compute_crop(w_in, h_in)

        # Ausgabe-Container anlegen
        with av.open(str(out_path), mode="w") as out_ctr:
            v_out = out_ctr.add_stream(vcodec, rate=rate)
            for k, v in vopts.items():
                setattr(v_out.codec_context, k, v) if hasattr(v_out.codec_context, k) else v_out.codec_context.options.update({k:v})
            v_out.pix_fmt = pixfmt
            v_out.width, v_out.height = cw, ch
            v_out.time_base = v_in.time_base or av.time_base

            a_out = None
            if a_in is not None:
                acodec, aopts = _audio_params()
                a_out = out_ctr.add_stream(acodec)
                for k, v in aopts.items():
                    if hasattr(a_out.codec_context, k):
                        setattr(a_out.codec_context, k, int(v) if v.isdigit() else v)
                    else:
                        a_out.codec_context.options.update({k:v})

            # Demux/Decode/Process/Encode
            frames_written = 0
            limit_pts = None
            if PREVIEW_SECONDS is not None and v_in.time_base:
                limit_pts = int(PREVIEW_SECONDS / v_in.time_base)

            # Fortschritt, wenn Framezahl bekannt
            total_frames = int(v_in.frames) if v_in.frames else 0
            bar = tqdm(total=total_frames if total_frames>0 and PREVIEW_SECONDS is None else 0,
                       unit="f", disable=(total_frames==0 or PREVIEW_SECONDS is not None))

            for packet in in_ctr.demux((v_in, a_in) if a_in is not None else (v_in,)):
                if packet.stream.type == "video":
                    for frame in packet.decode():
                        if limit_pts is not None and frame.pts is not None and frame.pts > limit_pts:
                            break
                        # in RGB konvertieren, croppen, neues Frame erzeugen
                        rgb = frame.to_ndarray(format="rgb24")
                        crop = rgb[cy:cy+ch, cx:cx+cw, :]
                        vf = av.VideoFrame.from_ndarray(crop, format="rgb24").reformat(width=cw, height=ch, format=pixfmt)
                        for pkt in v_out.encode(vf):
                            out_ctr.mux(pkt)
                        frames_written += 1
                        if total_frames: bar.update(1)
                elif a_in is not None and packet.stream.type == "audio":
                    # Audio decodieren und erneut kodieren (verlustfrei PCM oder AAC)
                    for afr in packet.decode():
                        for pkt in a_out.encode(afr):
                            out_ctr.mux(pkt)

            # Encoder flushen
            for pkt in v_out.encode(None):
                out_ctr.mux(pkt)
            if a_in is not None:
                for pkt in a_out.encode(None):
                    out_ctr.mux(pkt)

            if total_frames: bar.close()

    return out_path, (w_in, h_in), (cw, ch, cx, cy), frames_written, rate

def process_folder():
    vids = [p for p in INPUT_DIR.glob("*") if p.suffix.lower() in VIDEO_EXTS and p.is_file()]
    if not vids:
        print(f"Keine Videos in {INPUT_DIR.resolve()} gefunden.")
        return
    print(f"{len(vids)} Datei(en) gefunden. Starte Cropping mit QUALITY_MODE='{QUALITY_MODE}' …\n")
    for v in sorted(vids):
        try:
            outp, in_sz, crop, frames, fps = crop_video(v)
            (w,h),(cw,ch,cx,cy) = in_sz, crop
            print(f"{v.name}: {w}x{h} → crop {cw}x{ch}+{cx}+{cy}, {frames} Frames @ {fps:.2f} fps → {outp.name}")
        except Exception as e:
            print(f"Übersprungen ({v.name}): {e}")
    print("\nFertig. Ausgaben liegen in:", OUTPUT_DIR.resolve())

process_folder()
