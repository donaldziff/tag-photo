import hashlib
import json
import os
import sqlite3
import subprocess


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
    """)
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
    """Return sorted list of absolute TIFF paths in dir_path."""
    try:
        entries = os.listdir(dir_path)
    except FileNotFoundError:
        return []
    return sorted(
        os.path.join(dir_path, e)
        for e in entries
        if e.lower().endswith((".tif", ".tiff"))
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


def get_scans_for_dir(conn, scan_dir, pending_only=False):
    """Return recto scan records for scan_dir, ordered by rowid."""
    if pending_only:
        return conn.execute(
            "SELECT * FROM scans WHERE scan_dir=? AND is_verso=0 AND state='PENDING'"
            " ORDER BY rowid",
            (scan_dir,),
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


# ---------------------------------------------------------------------------
# Scan directory registration
# ---------------------------------------------------------------------------

def scan_directory(conn, archive_root, scan_dir, envelope_id=None):
    """Hash all TIFFs in scan_dir and register new ones as PENDING.

    If envelope_id is given it is set on newly inserted records only.
    Returns the number of newly added records.
    """
    dir_path = os.path.join(archive_root, scan_dir)
    added = 0
    for path in _find_tiffs(dir_path):
        file_hash = hash_file(path)
        filename = os.path.basename(path)
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
                 envelope_id=None):
    """Save metadata, write EXIF to the TIFF, and mark the photo REVIEWED.

    If the photo was EXPORTED or UPLOADED, clears jpeg_path and uploaded_at.
    """
    scan = get_scan(conn, hash_val)
    was_exported = scan["state"] in ("EXPORTED", "UPLOADED")

    conn.execute("""
        UPDATE scans SET
            description=?, verso_text=?, recto_stamp_text=?,
            date_inferred=?, date_source=?, envelope_id=?,
            state='REVIEWED', uploaded_at=NULL
        WHERE hash=?
    """, (description, verso_text, recto_stamp_text,
          date_inferred, date_source, envelope_id, hash_val))
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
    """Drop a photo back to REVIEWED and clear export state (browse-mode edit)."""
    scan = get_scan(conn, hash_val)
    if scan["state"] in ("EXPORTED", "UPLOADED"):
        conn.execute(
            "UPDATE scans SET state='REVIEWED', uploaded_at=NULL, jpeg_path=NULL WHERE hash=?",
            (hash_val,),
        )
        conn.commit()
