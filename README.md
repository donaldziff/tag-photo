# tag_photo

An interactive CLI tool for tagging scanned physical photographs with EXIF metadata. Designed for working through shoeboxes of old prints — it watches a directory for new scans, opens each one in Preview, prompts you for text written on the back (verso text), and uses an LLM to extract a date, which it writes directly to the file via exiftool.

Progress is tracked in a local SQLite database so you can stop and resume at any time.

## Use cases

**While scanning**: Drop photos onto your scanner in batches using Image Capture. The script watches the directory and presents each new scan as it arrives.

**Batch processing**: Point the script at an existing directory of TIFFs and work through them at your own pace.

## Requirements

- Python 3.9+
- [exiftool](https://exiftool.org): `brew install exiftool`
- [Ollama](https://ollama.com) (optional, for free local inference): `brew install ollama`
  - A local model: `ollama pull llama3.2`
- A [Gemini API key](https://aistudio.google.com/apikey) (optional, used as fallback or escalation)

At least one of Ollama or a Gemini API key is required.

## Installation

```bash
pip install google-genai
```

Set your Gemini API key if using it:
```bash
export GEMINI_API_KEY="your-key-here"
```

Optionally make the script globally accessible:
```bash
chmod +x tag_photo.py
ln -s /path/to/tag_photo.py /usr/local/bin/tag-photo
```

## Usage

```bash
# Run in current directory
tag-photo

# Run in a specific directory
tag-photo -d ~/Pictures/scans/grandmas-photos

# Use a specific local model
tag-photo -m mistral

# Skip local LLM, use Gemini directly
tag-photo --no-local

# Discard tracking database and start fresh (does not affect EXIF already written)
tag-photo --reset
```

## Workflow

On startup the script asks for a **baseline context** — a short description of the batch (e.g. "Grandma's photos, 1970s Cape Cod"). The baseline serves two purposes:

- Its text is written to `IPTC:Keywords` on every processed photo, making the batch searchable
- Its date (if any) is used as the default timestamp for photos with no verso text

After you enter the baseline, the script checks it with the LLM and shows the default timestamp it will infer. You can confirm or edit before any photos are processed.

For each photo:

1. The current EXIF date, description, and keywords are shown
2. The file opens in Preview (in the background — terminal keeps focus)
3. You type any text written on the back of the photo, or press Enter if there is none
4. If there is no verso text, the confirmed baseline timestamp is applied directly (no LLM call)
5. If there is verso text, the LLM extracts a date and shows its reasoning
6. You accept, escalate to Gemini (if using local model), edit manually, or skip
7. The metadata is written and immediately verified by reading it back with exiftool

**Commands at the verso text prompt:**

| Input | Action |
|-------|--------|
| *(any text)* | Parse with AI |
| Enter | Apply baseline date (no verso text) |
| `m` | Mark as MAYBE — come back later |
| `s` | Skip |
| `q` | Quit |

**Commands after seeing the AI result:**

| Input | Action |
|-------|--------|
| Enter | Accept |
| `p` | Retry with Gemini (only shown when using local model) |
| `e` | Edit timestamp or description manually |
| `s` | Skip |
| `q` | Quit |

## File states

Each scan is tracked in `.scans.db` with one of these states:

| State | Meaning |
|-------|---------|
| `PENDING` | Not yet processed |
| `MAYBE` | Flagged to revisit |
| `SKIPPED` | Deliberately skipped |
| `PROCESSED` | EXIF metadata written |
| `FAILED` | LLM or exiftool error |

## EXIF fields written

- `DateTimeOriginal` — date extracted or inferred from verso text and baseline
- `Description` — the verbatim verso text (blank if none)
- `IPTC:Keywords` — the verbatim baseline context (blank if no baseline was set)

## Date inference rules

The LLM follows these rules when extracting dates:

- Exact date found → `YYYY:MM:DD 12:00:00`
- Month and year only → `YYYY:MM:01 12:00:00`
- Year only → `YYYY:01:01 12:00:00`
- No date found → no `DateTimeOriginal` written
- US date format assumed (`MM-DD-YY`) unless context suggests otherwise
- Verso text takes priority over baseline if they conflict

## Running tests

```bash
pip install pytest
python -m pytest test_tag_photo.py -v
```
