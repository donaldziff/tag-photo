# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Overview

A two-phase tool for digitizing and tagging a personal photo archive:

1. **Scanning phase**: Use macOS Image Capture to scan photos into dated scan directories. No tagging happens at scan time.
2. **Tagging phase**: A local web UI for working through untagged photos, adding metadata, and writing EXIF.
3. **Export phase**: Generate JPEGs for reviewed photos into a per-scan-directory `export/` subdirectory, ready for manual upload to Google Photos.

The physical archive consists of numbered envelopes (e.g. `88`, `88p1`, `88p2`), each containing a small selection of photos chosen for scanning. Fewer than 10% of photos have useful verso text.

## Directory layout

Each family archive has its own root directory:

```
~/photos/familyname/
    tagger.db              ← single DB for the entire archive
    scans-2024-01/
        scan0001.tif
        scan0002.tif
        export/            ← generated JPEGs, one upload batch
    scans-2024-03/
        scan0001.tif       ← no conflict: different scan directory
        export/
    ...
```

Scan subdirectory names are date-based by convention (e.g. `scans-2024-01`). Each `export/` subdirectory corresponds to one Google Photos upload batch. Multiple archives are fully independent (separate root directories, separate databases).

## Physical scanning workflow

- Photos are scanned in batches using Image Capture's **Detect Separate Items** feature (4–5 at a time), producing sequentially named TIFFs in the current scan directory.
- Photos with interesting verso (handwriting, printed processor dates, photo lab stamps) are scanned individually as a normal sequential scan — no special Image Capture settings needed.
- Immediately after scanning a verso, run `manage.py mark-verso` to associate the two most recently added TIFFs as a recto/verso pair. The command shows both filenames and prompts for confirmation before committing.
- If a verso is not paired at scan time, it can be flagged as `NEEDS_PAIRING` in the tagging UI for later resolution.

## Upload workflow

Upload is manual — no API integration. When a scan directory's photos are sufficiently reviewed:

1. Run `manage.py export` to generate JPEGs into `scans-YYYY-MM/export/`
2. Upload the `export/` folder manually via the Google Photos UI
3. Run `manage.py mark-uploaded -d <scan_dir>` to record the upload in the DB
4. In Google Photos, create an album from the uploaded batch and archive it to remove it from the main timeline (this step cannot be automated via the Google Photos API)

## Commands

```bash
# Run all tests
python -m pytest -v

# Initialize a new archive
python manage.py init -a <archive_root>

# Import envelope list (one-time migration from envelopes.txt)
python manage.py import-envelopes -a <archive_root> envelopes.txt

# Register a new scan directory and hash its TIFFs into the DB
python manage.py scan-dir -a <archive_root> -d <scan_dir>

# Associate the two most recently added TIFFs as a recto/verso pair
python manage.py mark-verso -a <archive_root> -d <scan_dir>

# Generate JPEGs for all REVIEWED photos in a scan directory
python manage.py export -a <archive_root> -d <scan_dir>

# Mark all photos in a scan directory's export/ as uploaded
python manage.py mark-uploaded -a <archive_root> -d <scan_dir>

# Start the tagging web UI
python app.py -a <archive_root>

# Migration: scan an existing directory and import any existing EXIF
python manage.py migrate-dir -a <archive_root> -d <scan_dir>
```

## Architecture

**`app.py`**: Flask web application. Serves the tagging UI and exposes a small JSON API consumed by the frontend. Takes `-a <archive_root>`.

**`tagger.py`**: Core logic — queue management, LLM calls, EXIF reads/writes, JPEG export. Imported by `app.py` and testable independently.

**`manage.py`**: CLI for all administrative tasks (init, envelope management, scan directory registration, verso pairing, export, upload tracking, migration).

**`test_tagger.py`**: pytest suite covering tagger logic and LLM retry behavior.

## Data layer

SQLite database (`tagger.db`) lives at the archive root. The **primary key for every scan is a SHA-256 content hash** — not filename or path. This ensures stability across renames and enables duplicate detection.

**`envelopes`** table:
- `id` — string identifier matching physical envelope labels (e.g. `88`, `88p1`, `88p2`)
- `description` — short freeform description (e.g. "Mom's side, Cape Cod 1960s–70s")

**`scans`** table:
- `hash` — SHA-256 of file content (primary key)
- `filename` — original filename
- `scan_dir` — scan subdirectory (e.g. `scans-2024-01`)
- `is_verso` — boolean; verso images are not shown in the main tagging queue
- `verso_hash` — hash of paired verso scan, if any (FK to `scans.hash`; null otherwise)
- `envelope_id` — FK to envelopes (set at tagging time)
- `verso_text` — verbatim text from the verso (LLM-transcribed from verso image, or entered manually)
- `recto_stamp_text` — date or text printed on the recto border by the photo processor
- `description` — caption: who or what is in the photo
- `date_inferred` — best date estimate (YYYY, YYYY-MM, or YYYY-MM-DD)
- `date_source` — one of: `verso_text`, `verso_image`, `recto_stamp`, `llm_guess`, `manual`
- `state` — see State machine below
- `jpeg_path` — relative path to exported JPEG, if any
- `uploaded_at` — timestamp when marked as uploaded, null otherwise

## State machine

```
PENDING → REVIEWED → EXPORTED → UPLOADED
```

Side states:
- `SKIPPED` — not worth tagging; excluded from queue
- `NEEDS_PAIRING` — looks like an unpaired verso; flagged in tagging UI, resolved via `manage.py mark-verso` or UI, then returns to `PENDING`

Transitions:
- Any metadata edit on a `REVIEWED`, `EXPORTED`, or `UPLOADED` photo drops it back to `REVIEWED`, triggering JPEG regeneration on next export pass and clearing `uploaded_at`
- Verso images (`is_verso = TRUE`) never appear in the tagging queue or thumbnail grid; they are shown alongside their paired recto during review

## LLM layer

LLM backends are pluggable. Each backend is a callable with the signature:

```python
def llm_fn(prompt: str, image_path: str | None = None) -> str:
    # returns raw JSON text
```

Factory functions:
- `make_gemini_llm_fn()` — uses `GEMINI_API_KEY`
- `make_ollama_llm_fn()` — uses local Ollama at `http://localhost:11434`
- `make_anthropic_llm_fn()` — uses `ANTHROPIC_API_KEY`

`parse_with_llm(fn, prompt, image_path)` wraps any backend with retry logic for rate limits and malformed responses.

The LLM is used for two tasks:
1. **Verso transcription**: given a verso scan image, extract any text verbatim
2. **Date inference**: given verso text and/or recto stamp text, return a best-guess date with confidence

LLM inference runs on demand in the UI (button), not automatically.

## EXIF layer

All reads/writes go through `exiftool` via `subprocess`. `write_exif` sets:
- `DateTimeOriginal`, `IPTC:DateCreated`, `XMP:CreateDate` (all three together)
- `Description`
- `IPTC:Keywords` (includes envelope description)

After every write, the result is read back and verified.

## JPEG export

`manage.py export` generates JPEGs into `<scan_dir>/export/`:
- Quality: 90
- EXIF propagated from the source TIFF
- Rectos only (verso images are not exported)
- Only `REVIEWED` photos without an up-to-date JPEG are processed
- If a photo has been edited and dropped back to `REVIEWED`, its JPEG is regenerated

## Web UI

A Flask app with two modes, both scoped to a scan directory.

### Thumbnail grid

The entry point for both modes. Displays one thumbnail per recto in the scan directory (verso images are never shown directly).

- **Browse mode**: shows all photos regardless of state; each thumbnail displays a state badge, an envelope ID if assigned, and a verso indicator if paired
- **Tag mode**: shows only `PENDING` photos

Clicking any thumbnail navigates to the detail page for that photo.

### Detail page

The same page layout is used in both modes:

- Recto image (large)
- Verso image alongside, if paired
- All metadata fields: envelope, description, verso text, recto stamp text, date, date source
- EXIF data as currently written to the file (read back via exiftool)
- Prev/next arrows for sequential navigation within the scan directory (in the same mode's sequence)

**Tag mode**: fields are editable; LLM inference available on demand; actions include Accept (→ REVIEWED), Skip, and Flag as NEEDS_PAIRING.

**Browse mode**: fields are read-only. An explicit **Edit** button re-opens the photo for editing, dropping its state back to `REVIEWED`. This is the only way to trigger a state regression from browse mode.

## Envelope management

Envelopes are managed in the DB. `envelopes.txt` is a one-time migration input only — after import it is not kept in sync. New envelopes can be added via `manage.py add-envelope` or the tagging UI.

## Migration from previous workflow

Previously scanned photos may already have EXIF written by the old `tag_photo.py` tool. Run `manage.py migrate-dir` on each existing directory to:
- Hash each TIFF and register it in the DB
- Read back any existing EXIF into the appropriate DB fields
- Create records in state `REVIEWED` (not `PENDING` — they have already been through a workflow)

Migrated photos appear in browse mode and can be re-opened for editing if corrections are needed.

## External dependencies

- `exiftool` must be on PATH (`brew install exiftool`)
- At least one LLM backend configured (see LLM layer above)
- Python packages: `flask`, `google-genai`, `anthropic` (all optional depending on backend used)
