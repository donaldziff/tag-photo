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


def scan_directory(conn, archive_root, scan_dir):
    """Hash all TIFFs in scan_dir and register new ones as PENDING.

    Returns the number of newly added records.
    """
    dir_path = os.path.join(archive_root, scan_dir)
    added = 0
    for path in _find_tiffs(dir_path):
        file_hash = hash_file(path)
        filename = os.path.basename(path)
        cursor = conn.execute(
            "INSERT OR IGNORE INTO scans (hash, filename, scan_dir, state)"
            " VALUES (?, ?, ?, 'PENDING')",
            (file_hash, filename, scan_dir),
        )
        if cursor.rowcount > 0:
            added += 1
    conn.commit()
    return added
