import hashlib
import json
import os
import sqlite3
import subprocess
import time
from datetime import datetime, timezone


def init_db(archive_root):
    """Open (or create) tagger.db at archive_root and ensure schema exists."""
    db_path = os.path.join(archive_root, "tagger.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS envelopes (
            id          TEXT PRIMARY KEY,
            description TEXT
        );

        CREATE TABLE IF NOT EXISTS scans (
            hash             TEXT PRIMARY KEY,
            filename         TEXT NOT NULL,
            scan_dir         TEXT NOT NULL,
            is_verso         INTEGER NOT NULL DEFAULT 0,
            verso_hash       TEXT REFERENCES scans(hash),
            envelope_id      TEXT REFERENCES envelopes(id),
            verso_text       TEXT,
            recto_stamp_text TEXT,
            description      TEXT,
            date_inferred    TEXT,
            date_source      TEXT,
            state            TEXT NOT NULL DEFAULT 'PENDING',
            jpeg_path        TEXT,
            uploaded_at      TEXT
        );

        CREATE TABLE IF NOT EXISTS llm_usage (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            model         TEXT NOT NULL,
            input_tokens  INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL
        );
    """)
    try:
        conn.execute("ALTER TABLE scans ADD COLUMN date_confidence TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        conn.execute("ALTER TABLE scans ADD COLUMN subjects TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        conn.execute("ALTER TABLE scans ADD COLUMN asset_id TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        conn.execute("ALTER TABLE scans ADD COLUMN store_type TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # EXPORTED/UPLOADED are no longer states; they're derived from jpeg_path/asset_id.
    conn.execute("UPDATE scans SET state='REVIEWED' WHERE state IN ('EXPORTED', 'UPLOADED')")
    conn.commit()


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------

def hash_file(path):
    """Return the SHA-256 hex digest of the file at path."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_tiffs(dir_path):
    """Return sorted list of absolute TIFF/PNG paths in dir_path."""
    try:
        entries = os.listdir(dir_path)
    except FileNotFoundError:
        return []
    return sorted(
        os.path.join(dir_path, e)
        for e in entries
        if e.lower().endswith((".tif", ".tiff", ".png"))
    )


# ---------------------------------------------------------------------------
# EXIF read / write
# ---------------------------------------------------------------------------

def _date_to_exif_ts(date_str):
    """Convert YYYY, YYYY-MM, or YYYY-MM-DD to EXIF 'YYYY:MM:DD HH:MM:SS'."""
    if not date_str:
        return None
    parts = date_str.split("-")
    year = parts[0]
    month = parts[1] if len(parts) > 1 else "01"
    day   = parts[2] if len(parts) > 2 else "01"
    return f"{year}:{month}:{day} 12:00:00"


def read_exif(file_path):
    """Return dict of EXIF tags currently on file_path (empty dict on failure)."""
    result = subprocess.run(
        ["exiftool", "-json",
         "-DateTimeOriginal", "-IPTC:DateCreated", "-XMP:CreateDate",
         "-Description", "-IPTC:Keywords",
         file_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    records = json.loads(result.stdout)
    return records[0] if records else {}


def write_exif(file_path, date_inferred=None, description=None, envelope_description=None):
    """Write metadata to file_path via exiftool.

    date_inferred: YYYY, YYYY-MM, or YYYY-MM-DD
    """
    cmd = ["exiftool", "-overwrite_original"]
    ts = _date_to_exif_ts(date_inferred)
    if ts:
        cmd += [
            f"-DateTimeOriginal={ts}",
            f"-IPTC:DateCreated={ts}",
            f"-XMP:CreateDate={ts}",
        ]
    if description:
        cmd.append(f"-Description={description}")
    if envelope_description:
        cmd.append(f"-IPTC:Keywords={envelope_description}")
    cmd.append(file_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"exiftool failed: {result.stderr.strip()}")


# ---------------------------------------------------------------------------
# Image cache (thumbnails + web previews)
# ---------------------------------------------------------------------------

def _thumbs_dir(archive_root):
    return os.path.join(archive_root, ".thumbs")


def ensure_thumbnail(archive_root, hash_val, scan_dir, filename):
    """Return path to a cached JPEG thumbnail, generating it if needed."""
    from PIL import Image
    cache_path = os.path.join(_thumbs_dir(archive_root), f"{hash_val}_thumb.jpg")
    if not os.path.exists(cache_path):
        os.makedirs(_thumbs_dir(archive_root), exist_ok=True)
        img = Image.open(os.path.join(archive_root, scan_dir, filename))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((300, 400), Image.LANCZOS)
        img.save(cache_path, "JPEG", quality=75)
    return cache_path


def ensure_preview(archive_root, hash_val, scan_dir, filename):
    """Return path to a cached web-sized JPEG, generating it if needed."""
    from PIL import Image
    cache_path = os.path.join(_thumbs_dir(archive_root), f"{hash_val}_preview.jpg")
    if not os.path.exists(cache_path):
        os.makedirs(_thumbs_dir(archive_root), exist_ok=True)
        img = Image.open(os.path.join(archive_root, scan_dir, filename))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((1600, 1600), Image.LANCZOS)
        img.save(cache_path, "JPEG", quality=85)
    return cache_path


# ---------------------------------------------------------------------------
# DB queries
# ---------------------------------------------------------------------------

def get_scan(conn, hash_val):
    """Return a scan record as sqlite3.Row, or None."""
    return conn.execute("SELECT * FROM scans WHERE hash = ?", (hash_val,)).fetchone()


def get_envelope(conn, envelope_id):
    """Return an envelope record as sqlite3.Row, or None."""
    if not envelope_id:
        return None
    return conn.execute("SELECT * FROM envelopes WHERE id = ?", (envelope_id,)).fetchone()


def get_scans_unassigned(conn, state=None):
    """Return recto scans with no envelope assigned, optionally filtered by state."""
    if state:
        return conn.execute(
            "SELECT * FROM scans WHERE envelope_id IS NULL AND is_verso=0 AND state=?"
            " ORDER BY scan_dir, rowid",
            (state,),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM scans WHERE envelope_id IS NULL AND is_verso=0 ORDER BY scan_dir, rowid"
    ).fetchall()


def get_scans_for_envelope(conn, envelope_id, state=None):
    """Return recto scan records for envelope_id, ordered by scan_dir then rowid.

    If state is given, only scans in that state are returned.
    """
    if state:
        return conn.execute(
            "SELECT * FROM scans WHERE envelope_id=? AND is_verso=0 AND state=?"
            " ORDER BY scan_dir, rowid",
            (envelope_id, state),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM scans WHERE envelope_id=? AND is_verso=0 ORDER BY scan_dir, rowid",
        (envelope_id,),
    ).fetchall()


def get_scans_for_dir(conn, scan_dir, state=None):
    """Return recto scan records for scan_dir, ordered by rowid.

    If state is given, only scans in that state are returned.
    """
    if state:
        return conn.execute(
            "SELECT * FROM scans WHERE scan_dir=? AND is_verso=0 AND state=?"
            " ORDER BY rowid",
            (scan_dir, state),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM scans WHERE scan_dir=? AND is_verso=0 ORDER BY rowid",
        (scan_dir,),
    ).fetchall()


def get_scan_dirs(conn):
    """Return sorted list of distinct scan_dir values in the DB."""
    rows = conn.execute(
        "SELECT DISTINCT scan_dir FROM scans ORDER BY scan_dir"
    ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Envelope management
# ---------------------------------------------------------------------------

def upsert_envelope(conn, envelope_id, description=None):
    """Create envelope if it doesn't exist; update description if one is provided.

    Returns 'created', 'updated', or 'unchanged'.
    """
    existing = conn.execute(
        "SELECT description FROM envelopes WHERE id = ?", (envelope_id,)
    ).fetchone()

    if existing is None:
        conn.execute(
            "INSERT INTO envelopes (id, description) VALUES (?, ?)",
            (envelope_id, description),
        )
        conn.commit()
        return "created"

    if description is not None and description != existing[0]:
        conn.execute(
            "UPDATE envelopes SET description = ? WHERE id = ?",
            (description, envelope_id),
        )
        conn.commit()
        return "updated"

    return "unchanged"


def list_envelopes(conn):
    """Return list of (id, description, scan_count) sorted by id."""
    return conn.execute("""
        SELECT e.id, e.description, COUNT(s.hash) AS scan_count
        FROM envelopes e
        LEFT JOIN scans s ON s.envelope_id = e.id
        GROUP BY e.id
        ORDER BY e.id
    """).fetchall()


def get_envelopes_with_thumbnails(conn):
    """Return list of (id, description, scan_count, pending_count, sample_hash).

    scan_count counts rectos only, so envelopes containing only an unpaired
    verso are treated as empty (no browsable photos, no thumbnail).
    """
    rows = conn.execute("""
        SELECT e.id, e.description,
               COALESCE(SUM(CASE WHEN s.is_verso=0 THEN 1 ELSE 0 END), 0) AS scan_count,
               SUM(CASE WHEN s.state='PENDING' AND s.is_verso=0 THEN 1 ELSE 0 END) AS pending_count,
               MIN(CASE WHEN s.is_verso=0 THEN s.hash END) AS sample_hash
        FROM envelopes e
        LEFT JOIN scans s ON s.envelope_id = e.id
        GROUP BY e.id
        ORDER BY e.id
    """).fetchall()
    return rows


# ---------------------------------------------------------------------------
# Verso pairing
# ---------------------------------------------------------------------------

def get_recent_pair(conn, scan_dir):
    """Return ((recto_hash, recto_filename), (verso_hash, verso_filename)) for the
    two most recently registered scans in scan_dir, or None if fewer than 2 exist.

    The higher-rowid record is treated as the verso candidate (it was scanned last).
    """
    rows = conn.execute(
        "SELECT hash, filename FROM scans WHERE scan_dir = ? ORDER BY rowid DESC LIMIT 2",
        (scan_dir,),
    ).fetchall()
    if len(rows) < 2:
        return None
    verso = rows[0]   # most recently added
    recto = rows[1]
    return recto, verso


def set_verso_pair(conn, recto_hash, verso_hash):
    """Mark verso_hash as a verso and link it from recto_hash."""
    conn.execute("UPDATE scans SET is_verso = 1 WHERE hash = ?", (verso_hash,))
    conn.execute("UPDATE scans SET verso_hash = ? WHERE hash = ?", (verso_hash, recto_hash))
    conn.commit()


def swap_verso_pair(conn, recto_hash, verso_hash):
    """Swap which image is the recto and which is the verso."""
    conn.execute("UPDATE scans SET is_verso=0, verso_hash=? WHERE hash=?", (recto_hash, verso_hash))
    conn.execute("UPDATE scans SET is_verso=1, verso_hash=NULL WHERE hash=?", (recto_hash,))
    conn.commit()


def delete_scan(conn, archive_root, hash_val):
    """Delete a scan: removes the file, cached thumbnails, and DB record.
    If the scan has a paired verso, that verso is unlinked back to a regular pending scan."""
    scan = get_scan(conn, hash_val)
    if not scan:
        return
    if scan["verso_hash"]:
        conn.execute("UPDATE scans SET is_verso=0, verso_hash=NULL WHERE hash=?", (scan["verso_hash"],))
    for suffix in ("_thumb.jpg", "_preview.jpg"):
        cache = os.path.join(_thumbs_dir(archive_root), f"{hash_val}{suffix}")
        if os.path.exists(cache):
            os.remove(cache)
    file_path = os.path.join(archive_root, scan["scan_dir"], scan["filename"])
    if os.path.exists(file_path):
        os.remove(file_path)
    conn.execute("DELETE FROM scans WHERE hash=?", (hash_val,))


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def rotate_scan_file(conn, archive_root, hash_val, degrees):
    """Rotate a scan image by degrees (positive=CCW, negative=CW).
    Updates the DB hash, clears cache. Returns the new hash.
    """
    from PIL import Image
    scan = get_scan(conn, hash_val)
    file_path = os.path.join(archive_root, scan["scan_dir"], scan["filename"])

    img = Image.open(file_path)
    img.rotate(degrees, expand=True).save(file_path)

    new_hash = hash_file(file_path)

    for suffix in ("_thumb.jpg", "_preview.jpg"):
        cache = os.path.join(_thumbs_dir(archive_root), f"{hash_val}{suffix}")
        if os.path.exists(cache):
            os.remove(cache)

    conn.execute("UPDATE scans SET verso_hash=? WHERE verso_hash=?", (new_hash, hash_val))
    conn.execute("UPDATE scans SET hash=? WHERE hash=?", (new_hash, hash_val))
    conn.commit()
    return new_hash


# ---------------------------------------------------------------------------
# JPEG export
# ---------------------------------------------------------------------------

def export_scan(archive_root, scan):
    """Generate a full-resolution JPEG for one REVIEWED scan, copying EXIF from the TIFF.

    Returns the jpeg_path relative to archive_root.
    """
    from PIL import Image
    scan_dir = scan["scan_dir"]
    tiff_path = os.path.join(archive_root, scan_dir, scan["filename"])
    export_dir = os.path.join(archive_root, scan_dir, "export")
    os.makedirs(export_dir, exist_ok=True)

    base = os.path.splitext(scan["filename"])[0]
    jpeg_name = base + ".jpg"
    jpeg_path = os.path.join(export_dir, jpeg_name)

    if os.path.exists(jpeg_path) and os.path.getmtime(jpeg_path) >= os.path.getmtime(tiff_path):
        return os.path.join(scan_dir, "export", jpeg_name)

    img = Image.open(tiff_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(jpeg_path, "JPEG", quality=90)

    result = subprocess.run(
        ["exiftool", "-TagsFromFile", tiff_path, "-all:all",
         "-overwrite_original", jpeg_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"exiftool EXIF copy failed: {result.stderr.strip()}")

    return os.path.join(scan_dir, "export", jpeg_name)


def export_directory(conn, archive_root, scan_dir):
    """Export all REVIEWED rectos in scan_dir to JPEG. Returns count exported."""
    scans = conn.execute(
        "SELECT * FROM scans WHERE scan_dir=? AND is_verso=0 AND state='REVIEWED'",
        (scan_dir,),
    ).fetchall()
    exported = 0
    for scan in scans:
        jpeg_rel_path = export_scan(archive_root, scan)
        conn.execute(
            "UPDATE scans SET jpeg_path=? WHERE hash=?",
            (jpeg_rel_path, scan["hash"]),
        )
        exported += 1
    conn.commit()
    return exported


def mark_uploaded(conn, scan_dir):
    """Mark all exported (jpeg_path set), not-yet-uploaded scans in scan_dir as uploaded."""
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result = conn.execute(
        "UPDATE scans SET uploaded_at=? WHERE scan_dir=? AND jpeg_path IS NOT NULL AND uploaded_at IS NULL",
        (now, scan_dir),
    )
    conn.commit()
    return result.rowcount


# ---------------------------------------------------------------------------
# Migration from old tag_photo.py workflow
# ---------------------------------------------------------------------------

def _exif_ts_to_date(ts):
    """Convert EXIF timestamp '1989:08:19 12:00:00' to compressed date string.
    day==01, month==01 → YYYY; day==01 → YYYY-MM; otherwise → YYYY-MM-DD."""
    if not ts:
        return None
    date_part = ts.split(' ')[0]
    parts = date_part.split(':')
    if len(parts) < 2:
        return parts[0]
    year, month = parts[0], parts[1]
    day = parts[2] if len(parts) > 2 else '01'
    if day == '01' and month == '01':
        return year
    if day == '01':
        return f"{year}-{month}"
    return f"{year}-{month}-{day}"


def migrate_directory(conn, archive_root, scan_dir):
    """Import TIFFs from scan_dir into the new DB, reading metadata from the old .scans.db.

    Returns (added, skipped) counts.
    State mapping: PROCESSED → REVIEWED, SKIPPED → SKIPPED, anything else → PENDING.
    applied_description → verso_text, applied_timestamp → date_inferred (date_source=manual).
    """
    dir_path = os.path.join(archive_root, scan_dir)
    old_db_path = os.path.join(dir_path, '.scans.db')

    old_records = {}
    if os.path.exists(old_db_path):
        import sqlite3 as _sqlite3
        old_conn = _sqlite3.connect(old_db_path)
        old_conn.row_factory = _sqlite3.Row
        for row in old_conn.execute("SELECT filename, state, applied_timestamp, applied_description FROM scans"):
            old_records[row['filename']] = row
        old_conn.close()

    STATE_MAP = {'PROCESSED': 'REVIEWED', 'SKIPPED': 'SKIPPED'}
    added = skipped = 0

    for path in _find_tiffs(dir_path):
        filename = os.path.basename(path)
        file_hash = hash_file(path)

        existing = conn.execute("SELECT hash FROM scans WHERE hash=?", (file_hash,)).fetchone()
        if existing:
            skipped += 1
            continue

        old = old_records.get(filename)
        state = STATE_MAP.get(old['state'] if old else None, 'PENDING')
        verso_text = (old['applied_description'] or None) if old else None
        date_inferred = _exif_ts_to_date(old['applied_timestamp'] if old else None)
        date_source = 'manual' if date_inferred else None

        conn.execute(
            "INSERT INTO scans (hash, filename, scan_dir, verso_text, date_inferred, date_source, state)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (file_hash, filename, scan_dir, verso_text, date_inferred, date_source, state),
        )
        added += 1

    conn.commit()
    return added, skipped


# ---------------------------------------------------------------------------
# Scan directory registration
# ---------------------------------------------------------------------------

def scan_directory(conn, archive_root, scan_dir, envelope_id=None):
    """Hash all TIFFs/PNGs in scan_dir and register new ones as PENDING.

    Skips files whose filename is already registered in this scan_dir (prevents
    duplicates when a file is edited and re-hashed).
    Returns the number of newly added records.
    """
    dir_path = os.path.join(archive_root, scan_dir)
    existing_filenames = {
        row[0] for row in conn.execute(
            "SELECT filename FROM scans WHERE scan_dir=?", (scan_dir,)
        )
    }
    added = 0
    for path in _find_tiffs(dir_path):
        filename = os.path.basename(path)
        if filename in existing_filenames:
            continue
        file_hash = hash_file(path)
        cursor = conn.execute(
            "INSERT OR IGNORE INTO scans (hash, filename, scan_dir, envelope_id, state)"
            " VALUES (?, ?, ?, ?, 'PENDING')",
            (file_hash, filename, scan_dir, envelope_id),
        )
        if cursor.rowcount > 0:
            added += 1
    conn.commit()
    return added


# ---------------------------------------------------------------------------
# Tagging actions
# ---------------------------------------------------------------------------

def accept_photo(conn, hash_val, archive_root, description=None, verso_text=None,
                 recto_stamp_text=None, date_inferred=None, date_source=None,
                 date_confidence=None, envelope_id=None, subjects=None):
    """Save metadata, write EXIF to the TIFF, and mark the photo REVIEWED.

    If the photo had been exported, clears jpeg_path and uploaded_at.
    """
    scan = get_scan(conn, hash_val)
    was_exported = scan["jpeg_path"] is not None

    conn.execute("""
        UPDATE scans SET
            description=?, verso_text=?, recto_stamp_text=?,
            date_inferred=?, date_source=?, date_confidence=?, envelope_id=?,
            subjects=?,
            state='REVIEWED', uploaded_at=NULL
        WHERE hash=?
    """, (description, verso_text, recto_stamp_text,
          date_inferred, date_source, date_confidence, envelope_id, subjects, hash_val))
    if was_exported:
        conn.execute("UPDATE scans SET jpeg_path=NULL WHERE hash=?", (hash_val,))
    conn.commit()

    file_path = os.path.join(archive_root, scan["scan_dir"], scan["filename"])
    envelope_desc = None
    if envelope_id:
        env = get_envelope(conn, envelope_id)
        if env:
            envelope_desc = env["description"]
    write_exif(file_path, date_inferred=date_inferred, description=description,
               envelope_description=envelope_desc)


def set_scan_state(conn, hash_val, state):
    """Set the state of a scan record."""
    conn.execute("UPDATE scans SET state=? WHERE hash=?", (state, hash_val))
    conn.commit()


def reopen_photo(conn, hash_val):
    """Clear export/upload tracking for a REVIEWED photo (browse-mode edit)."""
    scan = get_scan(conn, hash_val)
    if scan["jpeg_path"] is not None:
        conn.execute(
            "UPDATE scans SET uploaded_at=NULL, jpeg_path=NULL WHERE hash=?",
            (hash_val,),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# LLM usage metering
# ---------------------------------------------------------------------------

# USD per million tokens. Update if Anthropic pricing changes; unknown
# models fall back to DEFAULT_PRICING_PER_MTOK.
PRICING_PER_MTOK = {
    "claude-sonnet-4-6": (3.00, 15.00),  # (input, output)
}
DEFAULT_PRICING_PER_MTOK = (3.00, 15.00)


def record_llm_usage(conn, model, input_tokens, output_tokens):
    """Record one LLM API call's token usage."""
    conn.execute(
        "INSERT INTO llm_usage (timestamp, model, input_tokens, output_tokens) VALUES (?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), model, input_tokens, output_tokens),
    )
    conn.commit()


def get_llm_usage_summary(conn):
    """Return aggregate LLM usage as {calls, input_tokens, output_tokens, cost_usd}.

    cost_usd is an estimate based on PRICING_PER_MTOK.
    """
    rows = conn.execute("""
        SELECT model,
               COUNT(*) AS calls,
               COALESCE(SUM(input_tokens), 0) AS input_tokens,
               COALESCE(SUM(output_tokens), 0) AS output_tokens
        FROM llm_usage
        GROUP BY model
    """).fetchall()

    cost_usd = 0.0
    for row in rows:
        in_price, out_price = PRICING_PER_MTOK.get(row["model"], DEFAULT_PRICING_PER_MTOK)
        cost_usd += row["input_tokens"] / 1_000_000 * in_price
        cost_usd += row["output_tokens"] / 1_000_000 * out_price

    return {
        "calls": sum(row["calls"] for row in rows),
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "cost_usd": cost_usd,
    }


# ---------------------------------------------------------------------------
# LLM inference
# ---------------------------------------------------------------------------

def make_anthropic_llm_fn(model="claude-sonnet-4-6", conn=None):
    import anthropic
    import base64
    client = anthropic.Anthropic()

    def llm_fn(prompt, image_path=None):
        content = []
        if image_path:
            with open(image_path, "rb") as f:
                data = base64.standard_b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(image_path)[1].lower().lstrip(".")
            media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                          "png": "image/png", "gif": "image/gif",
                          "webp": "image/webp"}.get(ext, "image/jpeg")
            content.append({"type": "image",
                             "source": {"type": "base64", "media_type": media_type, "data": data}})
        content.append({"type": "text", "text": prompt})
        msg = client.messages.create(
            model=model, max_tokens=1024,
            messages=[{"role": "user", "content": content}]
        )
        if conn is not None:
            record_llm_usage(conn, model, msg.usage.input_tokens, msg.usage.output_tokens)
        return msg.content[0].text

    return llm_fn


def parse_with_llm(llm_fn, prompt, image_path=None, max_retries=3):
    """Call llm_fn and return validated JSON text, retrying on rate limits or bad JSON."""
    for attempt in range(max_retries):
        try:
            raw = llm_fn(prompt, image_path)
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            json.loads(text)
            return text
        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                raise ValueError(f"LLM returned non-JSON after {max_retries} attempts: {raw[:300]}")
        except Exception as e:
            msg = str(e).lower()
            if ("rate" in msg or "overload" in msg) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("parse_with_llm: exhausted retries")


def extract_verso_text(llm_fn, image_path):
    """Return transcribed text from a verso scan image, or None if no text found."""
    prompt = (
        "This is a scan of the back of a physical photograph. "
        "Transcribe all text you can see, exactly as written — handwriting, printed dates, "
        "photo lab stamps, captions, and any other markings. "
        'Return JSON with a single key "verso_text" whose value is the transcribed text '
        "(preserve line breaks with \\n), or null if there is no text."
    )
    raw = parse_with_llm(llm_fn, prompt, image_path)
    return json.loads(raw).get("verso_text")


def infer_date(llm_fn, image_path=None, verso_text=None, recto_stamp_text=None):
    """Return {date, date_source, date_confidence} inferred from available clues."""
    parts = []
    if verso_text:
        parts.append(f"Text written/printed on back of photo:\n{verso_text}")
    if recto_stamp_text:
        parts.append(f"Text printed on front border by photo lab:\n{recto_stamp_text}")
    if image_path:
        parts.append(
            "An image of the photo itself is attached. If text clues don't pin down "
            "the date, estimate based on visual cues in the photo (clothing, hairstyles, "
            "photo paper/border style, color processing, image quality, etc.)."
        )
    prompt = (
        "Based on the following clues from a physical photograph, estimate when it was taken.\n\n"
        + "\n\n".join(parts)
        + "\n\nReturn JSON with:\n"
        '- "date": best estimate as YYYY, YYYY-MM, or YYYY-MM-DD (null if truly unknown)\n'
        '- "date_source": one of "verso_text", "recto_stamp", "visual_cues", "llm_guess"\n'
        '- "date_confidence": one of "high", "medium", "low"'
    )
    raw = parse_with_llm(llm_fn, prompt, image_path)
    result = json.loads(raw)
    return {
        "date": result.get("date"),
        "date_source": result.get("date_source"),
        "date_confidence": result.get("date_confidence"),
    }
