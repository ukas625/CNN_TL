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
