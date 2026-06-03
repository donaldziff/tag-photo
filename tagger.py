import hashlib
import os
import sqlite3


def init_db(archive_root):
    """Open (or create) tagger.db at archive_root and ensure schema exists."""
    db_path = os.path.join(archive_root, "tagger.db")
    conn = sqlite3.connect(db_path)
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
