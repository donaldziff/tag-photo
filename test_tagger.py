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


def test_scan_directory_sets_envelope_id():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.tif", b"img1")

        tagger.scan_directory(conn, archive, "scans-2024-01", envelope_id="88")

        envelope_id = conn.execute("SELECT envelope_id FROM scans").fetchone()[0]
        assert envelope_id == "88"
        conn.close()


def test_scan_directory_no_envelope_leaves_null():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.tif", b"img1")

        tagger.scan_directory(conn, archive, "scans-2024-01")

        envelope_id = conn.execute("SELECT envelope_id FROM scans").fetchone()[0]
        assert envelope_id is None
        conn.close()


def test_scan_directory_envelope_only_on_new_scans():
    """Re-running scan-dir with a different envelope must not overwrite existing records."""
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.tif", b"img1")

        tagger.scan_directory(conn, archive, "scans-2024-01", envelope_id="88")
        tagger.scan_directory(conn, archive, "scans-2024-01", envelope_id="99")

        envelope_id = conn.execute("SELECT envelope_id FROM scans").fetchone()[0]
        assert envelope_id == "88"
        conn.close()


def test_scan_directory_different_envelopes_per_batch():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.tif", b"img1")
        tagger.scan_directory(conn, archive, "scans-2024-01", envelope_id="88")

        make_tiff(os.path.join(d, "scans-2024-01"), "scan0002.tif", b"img2")
        tagger.scan_directory(conn, archive, "scans-2024-01", envelope_id="89")

        rows = conn.execute(
            "SELECT filename, envelope_id FROM scans ORDER BY filename"
        ).fetchall()
        assert rows == [("scan0001.tif", "88"), ("scan0002.tif", "89")]
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


# --- upsert_envelope ---

def test_upsert_envelope_creates_record():
    with tempfile.TemporaryDirectory() as d:
        conn = tagger.init_db(d)
        tagger.upsert_envelope(conn, "88")
        row = conn.execute("SELECT id, description FROM envelopes WHERE id = '88'").fetchone()
        assert row is not None
        assert row[0] == "88"
        conn.close()


def test_upsert_envelope_sets_description():
    with tempfile.TemporaryDirectory() as d:
        conn = tagger.init_db(d)
        tagger.upsert_envelope(conn, "88", "Cape Cod summer 1972")
        desc = conn.execute("SELECT description FROM envelopes WHERE id = '88'").fetchone()[0]
        assert desc == "Cape Cod summer 1972"
        conn.close()


def test_upsert_envelope_does_not_overwrite_existing_description():
    with tempfile.TemporaryDirectory() as d:
        conn = tagger.init_db(d)
        tagger.upsert_envelope(conn, "88", "original description")
        tagger.upsert_envelope(conn, "88", "new description")
        desc = conn.execute("SELECT description FROM envelopes WHERE id = '88'").fetchone()[0]
        assert desc == "original description"
        conn.close()


def test_upsert_envelope_is_idempotent():
    with tempfile.TemporaryDirectory() as d:
        conn = tagger.init_db(d)
        tagger.upsert_envelope(conn, "88", "Cape Cod")
        tagger.upsert_envelope(conn, "88", "Cape Cod")
        count = conn.execute("SELECT COUNT(*) FROM envelopes WHERE id = '88'").fetchone()[0]
        assert count == 1
        conn.close()


# --- get_recent_pair / set_verso_pair ---

def _seed_pair(conn, archive, scan_dir):
    """Insert two scans into scan_dir and return (recto_hash, verso_hash)."""
    sd_path = os.path.join(archive, scan_dir)
    recto_path = make_tiff(sd_path, "scan0015.tif", b"recto content")
    tagger.scan_directory(conn, archive, scan_dir)
    verso_path = make_tiff(sd_path, "scan0016.tif", b"verso content")
    tagger.scan_directory(conn, archive, scan_dir)
    return tagger.hash_file(recto_path), tagger.hash_file(verso_path)


def test_get_recent_pair_returns_none_for_empty_dir():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        assert tagger.get_recent_pair(conn, "scans-2024-01") is None
        conn.close()


def test_get_recent_pair_returns_none_for_single_scan():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.tif", b"img1")
        tagger.scan_directory(conn, archive, "scans-2024-01")
        assert tagger.get_recent_pair(conn, "scans-2024-01") is None
        conn.close()


def test_get_recent_pair_last_added_is_verso_candidate():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        recto_hash, verso_hash = _seed_pair(conn, archive, "scans-2024-01")

        pair = tagger.get_recent_pair(conn, "scans-2024-01")

        assert pair is not None
        (got_recto_hash, recto_fn), (got_verso_hash, verso_fn) = pair
        assert got_recto_hash == recto_hash
        assert got_verso_hash == verso_hash
        assert recto_fn == "scan0015.tif"
        assert verso_fn == "scan0016.tif"
        conn.close()


def test_get_recent_pair_scoped_to_scan_dir():
    """A scan in a different directory must not appear in the pair."""
    with tempfile.TemporaryDirectory() as d:
        for sd in ["scans-2024-01", "scans-2024-02"]:
            os.makedirs(os.path.join(d, sd))
        conn = tagger.init_db(d)
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.tif", b"other dir")
        tagger.scan_directory(conn, d, "scans-2024-01")
        make_tiff(os.path.join(d, "scans-2024-02"), "scan0015.tif", b"recto")
        tagger.scan_directory(conn, d, "scans-2024-02")
        make_tiff(os.path.join(d, "scans-2024-02"), "scan0016.tif", b"verso")
        tagger.scan_directory(conn, d, "scans-2024-02")

        pair = tagger.get_recent_pair(conn, "scans-2024-02")

        assert pair is not None
        (_, recto_fn), (_, verso_fn) = pair
        assert recto_fn == "scan0015.tif"
        assert verso_fn == "scan0016.tif"
        conn.close()


def test_set_verso_pair_marks_is_verso():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        recto_hash, verso_hash = _seed_pair(conn, archive, "scans-2024-01")

        tagger.set_verso_pair(conn, recto_hash, verso_hash)

        is_verso = conn.execute(
            "SELECT is_verso FROM scans WHERE hash = ?", (verso_hash,)
        ).fetchone()[0]
        assert is_verso == 1
        conn.close()


def test_set_verso_pair_recto_not_marked_verso():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        recto_hash, verso_hash = _seed_pair(conn, archive, "scans-2024-01")

        tagger.set_verso_pair(conn, recto_hash, verso_hash)

        is_verso = conn.execute(
            "SELECT is_verso FROM scans WHERE hash = ?", (recto_hash,)
        ).fetchone()[0]
        assert is_verso == 0
        conn.close()


def test_set_verso_pair_links_verso_hash_on_recto():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        recto_hash, verso_hash = _seed_pair(conn, archive, "scans-2024-01")

        tagger.set_verso_pair(conn, recto_hash, verso_hash)

        linked = conn.execute(
            "SELECT verso_hash FROM scans WHERE hash = ?", (recto_hash,)
        ).fetchone()[0]
        assert linked == verso_hash
        conn.close()


def test_set_verso_pair_recto_has_no_verso_hash_on_itself():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        recto_hash, verso_hash = _seed_pair(conn, archive, "scans-2024-01")

        tagger.set_verso_pair(conn, recto_hash, verso_hash)

        linked = conn.execute(
            "SELECT verso_hash FROM scans WHERE hash = ?", (verso_hash,)
        ).fetchone()[0]
        assert linked is None
        conn.close()
