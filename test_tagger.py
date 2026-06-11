import hashlib
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

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
        assert rows[0]["filename"] == "scan0001.tif"
        assert rows[0]["scan_dir"] == "scans-2024-01"
        assert rows[0]["state"] == "PENDING"
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
        assert rows[0]["filename"] == "scan0001.tif"
        assert rows[0]["envelope_id"] == "88"
        assert rows[1]["filename"] == "scan0002.tif"
        assert rows[1]["envelope_id"] == "89"
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
        for name in ["photo.jpg", "notes.txt"]:
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


def test_scan_directory_handles_png_extension():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.png", b"img1")

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


def test_upsert_envelope_updates_existing_description():
    with tempfile.TemporaryDirectory() as d:
        conn = tagger.init_db(d)
        tagger.upsert_envelope(conn, "88", "original description")
        tagger.upsert_envelope(conn, "88", "corrected description")
        desc = conn.execute("SELECT description FROM envelopes WHERE id = '88'").fetchone()[0]
        assert desc == "corrected description"
        conn.close()


def test_upsert_envelope_returns_created():
    with tempfile.TemporaryDirectory() as d:
        conn = tagger.init_db(d)
        result = tagger.upsert_envelope(conn, "88", "Cape Cod")
        assert result == "created"
        conn.close()


def test_upsert_envelope_returns_updated():
    with tempfile.TemporaryDirectory() as d:
        conn = tagger.init_db(d)
        tagger.upsert_envelope(conn, "88", "original")
        result = tagger.upsert_envelope(conn, "88", "updated")
        assert result == "updated"
        conn.close()


def test_upsert_envelope_returns_unchanged_when_same_desc():
    with tempfile.TemporaryDirectory() as d:
        conn = tagger.init_db(d)
        tagger.upsert_envelope(conn, "88", "Cape Cod")
        result = tagger.upsert_envelope(conn, "88", "Cape Cod")
        assert result == "unchanged"
        conn.close()


def test_upsert_envelope_returns_unchanged_when_no_desc():
    with tempfile.TemporaryDirectory() as d:
        conn = tagger.init_db(d)
        tagger.upsert_envelope(conn, "88", "Cape Cod")
        result = tagger.upsert_envelope(conn, "88")
        assert result == "unchanged"
        conn.close()


def test_upsert_envelope_is_idempotent():
    with tempfile.TemporaryDirectory() as d:
        conn = tagger.init_db(d)
        tagger.upsert_envelope(conn, "88", "Cape Cod")
        tagger.upsert_envelope(conn, "88", "Cape Cod")
        count = conn.execute("SELECT COUNT(*) FROM envelopes WHERE id = '88'").fetchone()[0]
        assert count == 1
        conn.close()


def test_list_envelopes_returns_all():
    with tempfile.TemporaryDirectory() as d:
        conn = tagger.init_db(d)
        tagger.upsert_envelope(conn, "88", "Cape Cod")
        tagger.upsert_envelope(conn, "89", "Budapest")
        rows = tagger.list_envelopes(conn)
        assert len(rows) == 2
        assert rows[0][0] == "88"
        assert rows[1][0] == "89"
        conn.close()


def test_list_envelopes_includes_scan_count():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        tagger.upsert_envelope(conn, "88", "Cape Cod")
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.tif", b"img1")
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0002.tif", b"img2")
        tagger.scan_directory(conn, archive, "scans-2024-01", envelope_id="88")
        rows = tagger.list_envelopes(conn)
        assert rows[0][2] == 2  # scan_count
        conn.close()


def test_list_envelopes_zero_count_for_empty_envelope():
    with tempfile.TemporaryDirectory() as d:
        conn = tagger.init_db(d)
        tagger.upsert_envelope(conn, "88", "Cape Cod")
        rows = tagger.list_envelopes(conn)
        assert rows[0][2] == 0
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


# --- _date_to_exif_ts ---

def test_date_to_exif_ts_year_only():
    assert tagger._date_to_exif_ts("1987") == "1987:01:01 12:00:00"


def test_date_to_exif_ts_year_month():
    assert tagger._date_to_exif_ts("1987-06") == "1987:06:01 12:00:00"


def test_date_to_exif_ts_full_date():
    assert tagger._date_to_exif_ts("1987-06-15") == "1987:06:15 12:00:00"


def test_date_to_exif_ts_none():
    assert tagger._date_to_exif_ts(None) is None


# --- write_exif ---

def test_write_exif_sets_all_date_fields():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        tagger.write_exif("/tmp/photo.tif", date_inferred="1987-06-15")
        args = mock_run.call_args[0][0]
        assert any("DateTimeOriginal=1987:06:15 12:00:00" in a for a in args)
        assert any("IPTC:DateCreated=1987:06:15 12:00:00" in a for a in args)
        assert any("XMP:CreateDate=1987:06:15 12:00:00" in a for a in args)


def test_write_exif_sets_description():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        tagger.write_exif("/tmp/photo.tif", description="Summer holiday")
        args = mock_run.call_args[0][0]
        assert any("Description=Summer holiday" in a for a in args)


def test_write_exif_sets_keywords_from_envelope():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        tagger.write_exif("/tmp/photo.tif", envelope_description="Cape Cod 1972")
        args = mock_run.call_args[0][0]
        assert any("Keywords=Cape Cod 1972" in a for a in args)


def test_write_exif_raises_on_failure():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="exiftool error")
        with pytest.raises(RuntimeError, match="exiftool failed"):
            tagger.write_exif("/tmp/photo.tif", date_inferred="1987")


def test_write_exif_skips_missing_fields():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        tagger.write_exif("/tmp/photo.tif")
        args = mock_run.call_args[0][0]
        assert not any("DateTimeOriginal" in a for a in args)
        assert not any("Description" in a for a in args)
        assert not any("Keywords" in a for a in args)


# --- read_exif ---

def test_read_exif_returns_fields():
    exif_json = json.dumps([{
        "SourceFile": "/tmp/photo.tif",
        "DateTimeOriginal": "1987:06:15 12:00:00",
        "Description": "Summer holiday",
    }])
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=exif_json)
        result = tagger.read_exif("/tmp/photo.tif")
    assert result["DateTimeOriginal"] == "1987:06:15 12:00:00"
    assert result["Description"] == "Summer holiday"


def test_read_exif_returns_empty_on_failure():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert tagger.read_exif("/tmp/photo.tif") == {}


# --- get_scan / get_scans_for_dir / get_scan_dirs ---

def test_get_scan_returns_record():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        path = make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.tif", b"img1")
        tagger.scan_directory(conn, archive, "scans-2024-01")
        h = tagger.hash_file(path)
        scan = tagger.get_scan(conn, h)
        assert scan is not None
        assert scan["filename"] == "scan0001.tif"
        conn.close()


def test_get_scan_returns_none_for_unknown():
    with tempfile.TemporaryDirectory() as d:
        conn = tagger.init_db(d)
        assert tagger.get_scan(conn, "deadbeef") is None
        conn.close()


def test_get_scans_for_dir_excludes_versos():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        recto_hash, verso_hash = _seed_pair(conn, archive, "scans-2024-01")
        tagger.set_verso_pair(conn, recto_hash, verso_hash)
        scans = tagger.get_scans_for_dir(conn, "scans-2024-01")
        assert len(scans) == 1
        assert scans[0]["hash"] == recto_hash
        conn.close()


def test_get_scans_for_dir_pending_only():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        for i, content in enumerate([b"a", b"b", b"c"]):
            make_tiff(os.path.join(d, "scans-2024-01"), f"scan{i:04d}.tif", content)
        tagger.scan_directory(conn, archive, "scans-2024-01")
        all_scans = tagger.get_scans_for_dir(conn, "scans-2024-01")
        conn.execute("UPDATE scans SET state='REVIEWED' WHERE hash=?", (all_scans[0]["hash"],))
        pending = tagger.get_scans_for_dir(conn, "scans-2024-01", pending_only=True)
        assert len(pending) == 2
        conn.close()


def test_get_scan_dirs_returns_distinct_dirs():
    with tempfile.TemporaryDirectory() as d:
        for sd in ["scans-2024-01", "scans-2024-02"]:
            os.makedirs(os.path.join(d, sd))
        conn = tagger.init_db(d)
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.tif", b"a")
        tagger.scan_directory(conn, d, "scans-2024-01")
        make_tiff(os.path.join(d, "scans-2024-02"), "scan0001.tif", b"b")
        tagger.scan_directory(conn, d, "scans-2024-02")
        dirs = tagger.get_scan_dirs(conn)
        assert dirs == ["scans-2024-01", "scans-2024-02"]
        conn.close()


# --- set_scan_state / reopen_photo ---

def test_set_scan_state():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.tif", b"img1")
        tagger.scan_directory(conn, archive, "scans-2024-01")
        h = conn.execute("SELECT hash FROM scans").fetchone()[0]
        tagger.set_scan_state(conn, h, "SKIPPED")
        assert conn.execute("SELECT state FROM scans WHERE hash=?", (h,)).fetchone()[0] == "SKIPPED"
        conn.close()


def test_reopen_photo_clears_export_state():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.tif", b"img1")
        tagger.scan_directory(conn, archive, "scans-2024-01")
        h = conn.execute("SELECT hash FROM scans").fetchone()[0]
        conn.execute("UPDATE scans SET state='EXPORTED', jpeg_path='export/x.jpg' WHERE hash=?", (h,))
        conn.commit()
        tagger.reopen_photo(conn, h)
        row = conn.execute("SELECT state, jpeg_path, uploaded_at FROM scans WHERE hash=?", (h,)).fetchone()
        assert row["state"] == "REVIEWED"
        assert row["jpeg_path"] is None
        assert row["uploaded_at"] is None
        conn.close()


def test_reopen_photo_noop_for_pending():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.tif", b"img1")
        tagger.scan_directory(conn, archive, "scans-2024-01")
        h = conn.execute("SELECT hash FROM scans").fetchone()[0]
        tagger.reopen_photo(conn, h)
        state = conn.execute("SELECT state FROM scans WHERE hash=?", (h,)).fetchone()[0]
        assert state == "PENDING"
        conn.close()


# --- accept_photo ---

def test_accept_photo_saves_metadata_and_sets_reviewed():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.tif", b"img1")
        tagger.scan_directory(conn, archive, "scans-2024-01")
        h = conn.execute("SELECT hash FROM scans").fetchone()[0]

        with patch("tagger.write_exif"):
            tagger.accept_photo(conn, h, archive,
                                description="Family at beach",
                                date_inferred="1972-07",
                                date_source="manual",
                                subjects="Mom, Dad, Charlie")

        row = conn.execute("SELECT * FROM scans WHERE hash=?", (h,)).fetchone()
        assert row["state"] == "REVIEWED"
        assert row["description"] == "Family at beach"
        assert row["date_inferred"] == "1972-07"
        assert row["date_source"] == "manual"
        assert row["subjects"] == "Mom, Dad, Charlie"
        conn.close()


def test_accept_photo_clears_jpeg_path_when_exported():
    with tempfile.TemporaryDirectory() as d:
        conn, archive = open_archive(d, "scans-2024-01")
        make_tiff(os.path.join(d, "scans-2024-01"), "scan0001.tif", b"img1")
        tagger.scan_directory(conn, archive, "scans-2024-01")
        h = conn.execute("SELECT hash FROM scans").fetchone()[0]
        conn.execute("UPDATE scans SET state='EXPORTED', jpeg_path='export/x.jpg' WHERE hash=?", (h,))
        conn.commit()

        with patch("tagger.write_exif"):
            tagger.accept_photo(conn, h, archive)

        row = conn.execute("SELECT jpeg_path FROM scans WHERE hash=?", (h,)).fetchone()
        assert row["jpeg_path"] is None
        conn.close()


# --- LLM usage metering ---

def test_get_llm_usage_summary_empty():
    with tempfile.TemporaryDirectory() as d:
        conn, _ = open_archive(d)
        summary = tagger.get_llm_usage_summary(conn)
        assert summary == {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        conn.close()


def test_record_llm_usage_accumulates():
    with tempfile.TemporaryDirectory() as d:
        conn, _ = open_archive(d)
        tagger.record_llm_usage(conn, "claude-sonnet-4-6", 100, 20)
        tagger.record_llm_usage(conn, "claude-sonnet-4-6", 50, 10)
        summary = tagger.get_llm_usage_summary(conn)
        assert summary == {"calls": 2, "input_tokens": 150, "output_tokens": 30}
        conn.close()


def test_make_anthropic_llm_fn_records_usage_when_conn_given():
    with tempfile.TemporaryDirectory() as d:
        conn, _ = open_archive(d)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"ok": true}')]
        mock_response.usage = MagicMock(input_tokens=123, output_tokens=45)
        mock_client.messages.create.return_value = mock_response

        with patch("anthropic.Anthropic", return_value=mock_client):
            llm_fn = tagger.make_anthropic_llm_fn(conn=conn)
            result = llm_fn("describe this photo")

        assert result == '{"ok": true}'
        summary = tagger.get_llm_usage_summary(conn)
        assert summary == {"calls": 1, "input_tokens": 123, "output_tokens": 45}
        conn.close()


def test_make_anthropic_llm_fn_skips_usage_without_conn():
    with tempfile.TemporaryDirectory() as d:
        conn, _ = open_archive(d)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"ok": true}')]
        mock_response.usage = MagicMock(input_tokens=123, output_tokens=45)
        mock_client.messages.create.return_value = mock_response

        with patch("anthropic.Anthropic", return_value=mock_client):
            llm_fn = tagger.make_anthropic_llm_fn()
            llm_fn("describe this photo")

        summary = tagger.get_llm_usage_summary(conn)
        assert summary == {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        conn.close()
