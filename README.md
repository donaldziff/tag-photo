# JFZ Photo Tagger

A two-phase tool for digitizing and tagging a personal photo archive.

1. **Scan** — use Image Capture to scan photos into a dated directory
2. **Tag** — a local web UI for reviewing photos, adding metadata, and writing EXIF
3. **Export** — generate JPEGs ready for manual upload to Google Photos

## Dependencies

```bash
brew install exiftool
pip install flask pillow google-genai anthropic
```

At least one LLM backend must be configured (see Tagging section).

## Setup

```bash
# Create the archive root (once per family archive)
python manage.py init -a ~/photos/jfz
```

## Scanning

### Image Capture settings

| Setting | Value |
|---|---|
| Kind | Color |
| Colors | Millions |
| Resolution | 600 dpi |
| Use Custom Size | ✓ 6.48 × 8.6 inches |
| Rotation Angle | -0.5 |
| Auto Selection | Detect Separate Items |
| Name | JFZ |
| Format | TIFF |
| Combine into single document | off |
| Image Correction | None |

Set **Scan To** to the scan directory for the session (e.g. `~/photos/jfz/jfzscans.20260603`). Create this directory before starting — Image Capture will not create it automatically.

### Session workflow

Create a dated scan directory and point Image Capture at it:

```bash
mkdir ~/photos/jfz/jfzscans.20260603
```

For each envelope, scan its photos with Image Capture, then immediately register them:

```bash
python manage.py scan-dir -a ~/photos/jfz -d jfzscans.20260603 \
  -e 88 --envelope-desc "Cape Cod summer 1972"

# Next envelope (--envelope-desc only needed the first time an envelope is used)
python manage.py scan-dir -a ~/photos/jfz -d jfzscans.20260603 -e 89
```

`scan-dir` is safe to run repeatedly — it only registers files not already in the database, so it acts as the boundary between envelope batches.

If a photo has interesting verso text or a date stamp, scan the verso immediately after the recto, then pair them:

```bash
python manage.py scan-dir -a ~/photos/jfz -d jfzscans.20260603 -e 89
python manage.py mark-verso -a ~/photos/jfz -d jfzscans.20260603
```

`mark-verso` shows the two most recently registered scans and asks for confirmation before pairing.

## Tagging

```bash
python app.py -a ~/photos/jfz
```

Set at least one of these environment variables:

```bash
export GEMINI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

## Export and upload

```bash
# Generate JPEGs for all reviewed photos in a scan directory
python manage.py export -a ~/photos/jfz -d jfzscans.20260603

# Upload export/ manually via the Google Photos UI, then record it
python manage.py mark-uploaded -a ~/photos/jfz -d jfzscans.20260603
```

## Command reference

```bash
python manage.py init -a <archive>
python manage.py scan-dir -a <archive> -d <scan_dir> [-e <envelope_id>] [--envelope-desc <desc>]
python manage.py mark-verso -a <archive> -d <scan_dir>
python manage.py export -a <archive> -d <scan_dir>
python manage.py mark-uploaded -a <archive> -d <scan_dir>
python manage.py import-envelopes -a <archive> envelopes.txt
python app.py -a <archive>
python -m pytest -v
```
