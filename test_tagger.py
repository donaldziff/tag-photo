import hashlib
import os
import tempfile

import tagger


# --- helpers ---

def make_tiff(dir_path, filename, content=None):
    """Write a fake TIFF file and return its path."""
    path = os.path.join(dir_path, filename)
    with open(path, "wb") as f:
        f.write(content if content is not None else filename.encode())
    return path


def open_archive(tmp_dir, scan_dir=None):
    """Create an archive with an optional scan subdirectory; return (conn, archive_root)."""
    if scan_dir:
        os.makedirs(os.path.join(tmp_dir, scan_dir), exist_ok=True)
    conn = tagger.init_db(tmp_dir)
    return conn, tmp_dir


# --- init_db ---

def test_init_db_creates_envelopes_table():
    with tempfile.TemporaryDirectory() as d:
        conn = tagger.init_db(d)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert "envelopes" in tables
        conn.close()


def test_init_db_creates_scans_table():
    with tempfile.TemporaryDirectory() as d:
        conn = tagger.init_db(d)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert "scans" in tables
        conn.close()


def test_init_db_creates_tagger_db_file():
    with tempfile.TemporaryDirectory() as d:
        conn = tagger.init_db(d)
        conn.close()
        assert os.path.exists(os.path.join(d, "tagger.db"))


def test_init_db_is_idempotent():
    with tempfile.TemporaryDirectory() as d:
        tagger.init_db(d).close()
        tagger.init_db(d).close()  # must not raise


def test_init_db_scans_has_expected_columns():
    with tempfile.TemporaryDirectory() as d:
        conn = tagger.init_db(d)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(scans)")}
        expected = {
            "hash", "filename", "scan_dir", "is_verso", "verso_hash",
            "envelope_id", "verso_text", "recto_stamp_text", "description",
            "date_inferred", "date_source", "state", "jpeg_path", "uploaded_at",
        }
        assert expected <= cols
        conn.close()


# --- hash_file ---

def test_hash_file_matches_sha256():
    content = b"hello world"
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(content)
        path = f.name
    try:
        result = tagger.hash_file(path)
        assert result == hashlib.sha256(content).hexdigest()
    finally:
        os.unlink(path)


def test_hash_file_is_deterministic():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"test data")
        path = f.name
    try:
        assert tagger.hash_file(path) == tagger.hash_file(path)
    finally:
        os.unlink(path)


def test_hash_file_differs_for_different_content():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"aaa")
        path1 = f.name
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"bbb")
        path2 = f.name
    try:
        assert tagger.hash_file(path1) != tagger.hash_file(path2)
    finally:
        os.unlink(path1)
        os.unlink(path2)


# --- scan_directory ---

def test_scan_directory_adds_tiffs():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.tif", b"img1")
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0002.tif", b"img2")

        added = tagger.scan_directory(conn, archive, "scans-2024-01")

        assert added == 2
        rows = conn.execute(
            "SELECT filename, scan_dir, state FROM scans ORDER BY filename"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0] == ("scan0001.tif", "scans-2024-01", "PENDING")
        conn.close()


def test_scan_directory_uses_hash_as_pk():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        content = b"unique content"
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.tif", content)

        tagger.scan_directory(conn, archive, "scans-2024-01")

        expected_hash = hashlib.sha256(content).hexdigest()
        row = conn.execute("SELECT hash FROM scans WHERE filename = 'scan0001.tif'").fetchone()
        assert row[0] == expected_hash
        conn.close()


def test_scan_directory_is_idempotent():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.tif", b"img1")

        tagger.scan_directory(conn, archive, "scans-2024-01")
        tagger.scan_directory(conn, archive, "scans-2024-01")

        count = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        assert count == 1
        conn.close()


def test_scan_directory_preserves_existing_state():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.tif", b"img1")

        tagger.scan_directory(conn, archive, "scans-2024-01")
        conn.execute("UPDATE scans SET state = 'REVIEWED'")
        conn.commit()

        tagger.scan_directory(conn, archive, "scans-2024-01")

        state = conn.execute("SELECT state FROM scans").fetchone()[0]
        assert state == "REVIEWED"
        conn.close()


def test_scan_directory_ignores_non_tiff():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        for name in ["photo.jpg", "photo.png", "notes.txt"]:
            open(os.path.join(d, "scans-2024-01", name), "w").close()

        added = tagger.scan_directory(conn, archive, "scans-2024-01")

        assert added == 0
        conn.close()


def test_scan_directory_handles_tif_extension():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.tif", b"img1")

        added = tagger.scan_directory(conn, archive, "scans-2024-01")

        assert added == 1
        conn.close()


def test_scan_directory_handles_tiff_extension():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.tiff", b"img1")

        added = tagger.scan_directory(conn, archive, "scans-2024-01")

        assert added == 1
        conn.close()


def test_scan_directory_handles_uppercase_extension():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.TIFF", b"img1")

        added = tagger.scan_directory(conn, archive, "scans-2024-01")

        assert added == 1
        conn.close()


def test_scan_directory_returns_added_count():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        for i in range(3):
            make_tiff(os.path.join(d, "scans-2024-01"), f"scan{i:04d}.tif", f"img{i}".encode())

        added = tagger.scan_directory(conn, archive, "scans-2024-01")

        assert added == 3
        conn.close()


def test_scan_directory_skips_duplicate_content_across_dirs():
    """Two files with identical content → one DB record (hash collision = same photo)."""
    with tempfile.TemporaryDirectory() as d:
        for sd in ["scans-2024-01", "scans-2024-02"]:
            os.makedirs(os.path.join(d, sd))
        conn = tagger.init_db(d)
        same_content = b"identical"
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.tif", same_content)
        make_tiff(os.path.join(d, "scans-2024-02"), "scan0001.tif", same_content)

        added1 = tagger.scan_directory(conn, d, "scans-2024-01")
        added2 = tagger.scan_directory(conn, d, "scans-2024-02")

        assert added1 == 1
        assert added2 == 0  # duplicate hash ignored
        assert conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0] == 1
        conn.close()


def test_scan_directory_default_state_is_pending():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.tif", b"img1")

        tagger.scan_directory(conn, archive, "scans-2024-01")

        state = conn.execute("SELECT state FROM scans").fetchone()[0]
        assert state == "PENDING"
        conn.close()
