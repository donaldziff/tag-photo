import json
import sqlite3
import os
import tempfile
import time
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
from tag_photo import (
    init_db, sync_directory_to_db, parse_with_llm, write_exif,
    is_file_stable, check_ollama_available, make_ollama_llm_fn,
    format_timestamp, fmt_ts, read_exif, TIMESTAMP_RE,
    print_result, prompt_accept, prompt_edit, file_creation_time,
)


# --- Helpers ---

def make_llm_fn(response: dict):
    """Returns an llm_fn that always returns the given dict as JSON."""
    def llm_fn(prompt, system_instruction):
        return json.dumps(response)
    return llm_fn


def fake_urlopen(response_data: dict):
    """Returns a context manager mock that yields a response with the given JSON body."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(response_data).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# --- init_db ---

def test_init_db_creates_table():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        conn = init_db(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='scans'")
        assert cursor.fetchone() is not None
        conn.close()
    finally:
        os.unlink(db_path)


def test_init_db_is_idempotent():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        init_db(db_path).close()
        init_db(db_path).close()  # should not raise
    finally:
        os.unlink(db_path)


# --- sync_directory_to_db ---

def test_sync_adds_new_tiffs():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "photo1.tiff"), "w").close()
        open(os.path.join(d, "photo2.tiff"), "w").close()

        conn = init_db(":memory:")
        sync_directory_to_db(conn, d)

        cursor = conn.cursor()
        cursor.execute("SELECT filename, state FROM scans ORDER BY filename")
        rows = cursor.fetchall()
        assert rows == [("photo1.tiff", "PENDING"), ("photo2.tiff", "PENDING")]
        conn.close()


def test_sync_stores_file_creation_time():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "photo1.tiff")
        open(path, "w").close()

        conn = init_db(":memory:")
        sync_directory_to_db(conn, d)

        cursor = conn.cursor()
        cursor.execute("SELECT file_created_at FROM scans WHERE filename = 'photo1.tiff'")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] is not None
        assert row[0] > 0
        conn.close()


def test_file_creation_time_returns_float():
    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as f:
        path = f.name
    try:
        result = file_creation_time(path)
        assert isinstance(result, float)
        assert result > 0
    finally:
        os.unlink(path)


def test_sync_detects_uppercase_TIFF():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "photo.TIFF"), "w").close()

        conn = init_db(":memory:")
        sync_directory_to_db(conn, d)

        cursor = conn.cursor()
        cursor.execute("SELECT filename FROM scans")
        rows = cursor.fetchall()
        assert ("photo.TIFF",) in rows
        conn.close()


def test_sync_ignores_non_tiff_files():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "photo.jpg"), "w").close()
        open(os.path.join(d, "photo.png"), "w").close()

        conn = init_db(":memory:")
        sync_directory_to_db(conn, d)

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM scans")
        assert cursor.fetchone()[0] == 0
        conn.close()


def test_sync_does_not_duplicate_existing_files():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "photo1.tiff"), "w").close()

        conn = init_db(":memory:")
        sync_directory_to_db(conn, d)
        sync_directory_to_db(conn, d)

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM scans WHERE filename = 'photo1.tiff'")
        assert cursor.fetchone()[0] == 1
        conn.close()


def test_sync_preserves_existing_state():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "photo1.tiff"), "w").close()

        conn = init_db(":memory:")
        sync_directory_to_db(conn, d)

        cursor = conn.cursor()
        cursor.execute("UPDATE scans SET state = 'MAYBE' WHERE filename = 'photo1.tiff'")
        conn.commit()

        sync_directory_to_db(conn, d)

        cursor.execute("SELECT state FROM scans WHERE filename = 'photo1.tiff'")
        assert cursor.fetchone()[0] == "MAYBE"
        conn.close()


# --- is_file_stable ---

def test_is_file_stable_returns_false_for_missing_file():
    assert is_file_stable("/tmp/does_not_exist_xyz.tiff") is False


def test_is_file_stable_returns_false_for_empty_file():
    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as f:
        path = f.name
    try:
        assert is_file_stable(path) is False
    finally:
        os.unlink(path)


def test_is_file_stable_returns_false_for_recently_modified():
    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as f:
        f.write(b"data")
        path = f.name
    try:
        # File was just written, mtime is recent
        assert is_file_stable(path, min_age=60.0) is False
    finally:
        os.unlink(path)


def test_is_file_stable_returns_true_for_old_file():
    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as f:
        f.write(b"data")
        path = f.name
    try:
        # Backdate mtime by 10 seconds
        old_time = time.time() - 10
        os.utime(path, (old_time, old_time))
        assert is_file_stable(path, min_age=2.0) is True
    finally:
        os.unlink(path)


# --- check_ollama_available ---

def test_check_ollama_available_returns_true_when_model_present():
    response = {"models": [{"name": "llama3.2:latest"}]}
    with patch("urllib.request.urlopen", return_value=fake_urlopen(response)):
        assert check_ollama_available("llama3.2") is True


def test_check_ollama_available_returns_true_for_exact_name_match():
    response = {"models": [{"name": "llama3.2"}]}
    with patch("urllib.request.urlopen", return_value=fake_urlopen(response)):
        assert check_ollama_available("llama3.2") is True


def test_check_ollama_available_returns_false_when_model_absent():
    response = {"models": [{"name": "mistral:latest"}]}
    with patch("urllib.request.urlopen", return_value=fake_urlopen(response)):
        assert check_ollama_available("llama3.2") is False


def test_check_ollama_available_returns_false_when_ollama_not_running():
    import urllib.error
    with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError()):
        assert check_ollama_available("llama3.2") is False


def test_check_ollama_available_returns_false_on_any_exception():
    with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
        assert check_ollama_available("llama3.2") is False


# --- make_ollama_llm_fn ---

def test_make_ollama_llm_fn_returns_response_content():
    response = {"message": {"content": '{"timestamp": "2000:01:01 12:00:00", "description": "test", "summary": "found year"}'}}
    with patch("urllib.request.urlopen", return_value=fake_urlopen(response)):
        fn = make_ollama_llm_fn("llama3.2")
        result = fn("my prompt", "my system instruction")
    assert '"timestamp"' in result


def test_make_ollama_llm_fn_sends_correct_model():
    response = {"message": {"content": '{}'}}
    captured = {}

    def capture_urlopen(req, timeout=None):
        captured['payload'] = json.loads(req.data.decode())
        return fake_urlopen(response)

    with patch("urllib.request.urlopen", capture_urlopen):
        fn = make_ollama_llm_fn("mistral")
        fn("prompt", "system")

    assert captured['payload']['model'] == 'mistral'


def test_make_ollama_llm_fn_sets_json_format():
    response = {"message": {"content": '{}'}}
    captured = {}

    def capture_urlopen(req, timeout=None):
        captured['payload'] = json.loads(req.data.decode())
        return fake_urlopen(response)

    with patch("urllib.request.urlopen", capture_urlopen):
        fn = make_ollama_llm_fn("llama3.2")
        fn("prompt", "system")

    assert captured['payload']['format'] == 'json'


# --- parse_with_llm ---

def test_parse_returns_parsed_json():
    response = {"timestamp": "1987:12:25 12:00:00", "description": "Christmas", "summary": "exact date found"}
    result = parse_with_llm("Christmas 1987", "Family photos", make_llm_fn(response))
    assert result == response


def test_parse_null_timestamp():
    response = {"timestamp": None, "description": "Unknown", "summary": "no date found"}
    result = parse_with_llm("some photo", "", make_llm_fn(response))
    assert result["timestamp"] is None


def test_parse_retries_on_rate_limit():
    call_count = 0

    def flaky_llm(prompt, system):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("429 retry in 1s")
        return json.dumps({"timestamp": "2000:01:01 12:00:00", "description": "test", "summary": "test"})

    with patch("time.sleep"):
        result = parse_with_llm("text", "baseline", flaky_llm, max_retries=3)

    assert call_count == 3
    assert result["timestamp"] == "2000:01:01 12:00:00"


def test_parse_raises_after_max_retries():
    def always_rate_limited(prompt, system):
        raise Exception("429 retry in 1s")

    with patch("time.sleep"):
        with pytest.raises(Exception, match="429"):
            parse_with_llm("text", "baseline", always_rate_limited, max_retries=3)


def test_parse_raises_immediately_on_non_rate_limit_error():
    call_count = 0

    def broken_llm(prompt, system):
        nonlocal call_count
        call_count += 1
        raise ValueError("bad credentials")

    with pytest.raises(ValueError, match="bad credentials"):
        parse_with_llm("text", "baseline", broken_llm, max_retries=3)

    assert call_count == 1


def test_parse_uses_retry_wait_time_from_error():
    delays = []

    def rate_limited_once(prompt, system):
        if not delays:
            raise Exception("429 retry in 5.0s")
        return json.dumps({"timestamp": None, "description": "", "summary": ""})

    def fake_sleep(seconds):
        delays.append(seconds)

    with patch("time.sleep", fake_sleep):
        parse_with_llm("text", "baseline", rate_limited_once, max_retries=2)

    assert delays == [6.0]


def test_parse_rejects_placeholder_timestamp():
    """Local models sometimes echo 'YYYY:01:01 12:00:00' literally — should be rejected."""
    call_count = 0

    def placeholder_then_good(prompt, system):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return json.dumps({"timestamp": "YYYY:01:01 12:00:00", "summary": "placeholder"})
        return json.dumps({"timestamp": "1992:01:01 12:00:00", "summary": "fixed"})

    result = parse_with_llm("text", "JFZ 1992", placeholder_then_good, max_retries=3)
    assert call_count == 2
    assert result["timestamp"] == "1992:01:01 12:00:00"


def test_parse_raises_after_all_retries_with_bad_timestamp():
    def always_bad(prompt, system):
        return json.dumps({"timestamp": "YYYY:01:01 12:00:00", "summary": "placeholder"})

    with pytest.raises(ValueError, match="Invalid timestamp"):
        parse_with_llm("text", "baseline", always_bad, max_retries=3)


def test_timestamp_re_accepts_valid():
    assert TIMESTAMP_RE.match("1992:01:01 12:00:00")
    assert TIMESTAMP_RE.match("2024:12:31 23:59:59")

def test_timestamp_re_rejects_placeholder():
    assert not TIMESTAMP_RE.match("YYYY:01:01 12:00:00")
    assert not TIMESTAMP_RE.match("1992:MM:01 12:00:00")


# --- write_exif ---

def test_write_exif_with_timestamp_and_description():
    with patch("subprocess.run") as mock_run:
        write_exif("/tmp/photo.tiff", "1987:12:25 12:00:00", "Christmas morning")
        mock_run.assert_called_once_with(
            ["exiftool", "-overwrite_original",
             "-DateTimeOriginal=1987:12:25 12:00:00",
             "-IPTC:DateCreated=1987:12:25 12:00:00",
             "-XMP:CreateDate=1987:12:25 12:00:00",
             "-Description=Christmas morning",
             "/tmp/photo.tiff"],
            stdout=-3
        )


def test_write_exif_date_fields_all_written_together():
    with patch("subprocess.run") as mock_run:
        write_exif("/tmp/photo.tiff", "1987:12:25 12:00:00", "")
        args = mock_run.call_args[0][0]
        assert any("DateTimeOriginal=1987:12:25 12:00:00" in a for a in args)
        assert any("IPTC:DateCreated=1987:12:25 12:00:00" in a for a in args)
        assert any("XMP:CreateDate=1987:12:25 12:00:00" in a for a in args)


def test_write_exif_date_fields_all_skipped_when_no_timestamp():
    with patch("subprocess.run") as mock_run:
        write_exif("/tmp/photo.tiff", None, "")
        args = mock_run.call_args[0][0]
        assert not any("DateTimeOriginal" in a for a in args)
        assert not any("DateCreated" in a for a in args)
        assert not any("CreateDate" in a for a in args)


def test_write_exif_writes_keywords():
    with patch("subprocess.run") as mock_run:
        write_exif("/tmp/photo.tiff", None, "", keywords="Grandma Cape Cod 1970s")
        args = mock_run.call_args[0][0]
        assert any("Keywords=Grandma Cape Cod 1970s" in a for a in args)


def test_write_exif_skips_none_keywords():
    with patch("subprocess.run") as mock_run:
        write_exif("/tmp/photo.tiff", None, "", keywords=None)
        args = mock_run.call_args[0][0]
        assert not any("Keywords" in a for a in args)


def test_write_exif_skips_missing_timestamp():
    with patch("subprocess.run") as mock_run:
        write_exif("/tmp/photo.tiff", None, "Christmas morning")
        args = mock_run.call_args[0][0]
        assert not any("DateTimeOriginal" in a for a in args)
        assert any("Description" in a for a in args)


def test_write_exif_skips_missing_description():
    with patch("subprocess.run") as mock_run:
        write_exif("/tmp/photo.tiff", "1987:12:25 12:00:00", "")
        args = mock_run.call_args[0][0]
        assert any("DateTimeOriginal" in a for a in args)
        assert not any("Description" in a for a in args)


def test_write_exif_skips_both_when_empty():
    with patch("subprocess.run") as mock_run:
        write_exif("/tmp/photo.tiff", None, "")
        args = mock_run.call_args[0][0]
        assert args == ["exiftool", "-overwrite_original", "/tmp/photo.tiff"]


# --- format_timestamp ---

def test_format_timestamp_full_date():
    assert format_timestamp("1987:06:15 12:00:00") == "June 15, 1987 12:00pm"

def test_format_timestamp_morning():
    assert format_timestamp("2001:03:04 09:23:00") == "March 4, 2001 9:23am"

def test_format_timestamp_midnight():
    assert format_timestamp("2000:01:01 00:00:00") == "January 1, 2000 12:00am"

def test_format_timestamp_none_returns_none():
    assert format_timestamp(None) is None

def test_format_timestamp_invalid_returns_original():
    assert format_timestamp("not-a-date") == "not-a-date"


# --- read_exif ---

def test_read_exif_returns_values():
    exiftool_json = json.dumps([{"DateTimeOriginal": "1987:06:15 12:00:00", "Description": "Summer holiday", "Keywords": "Cape Cod"}])
    mock_result = MagicMock(returncode=0, stdout=exiftool_json)
    with patch("subprocess.run", return_value=mock_result):
        ts, desc, kw = read_exif("/tmp/photo.tiff")
    assert ts == "1987:06:15 12:00:00"
    assert desc == "Summer holiday"
    assert kw == "Cape Cod"

def test_read_exif_returns_nones_when_tags_absent():
    exiftool_json = json.dumps([{"SourceFile": "/tmp/photo.tiff"}])
    mock_result = MagicMock(returncode=0, stdout=exiftool_json)
    with patch("subprocess.run", return_value=mock_result):
        ts, desc, kw = read_exif("/tmp/photo.tiff")
    assert ts is None
    assert desc is None
    assert kw is None

def test_read_exif_returns_nones_on_exiftool_failure():
    mock_result = MagicMock(returncode=1, stdout="")
    with patch("subprocess.run", return_value=mock_result):
        ts, desc, kw = read_exif("/tmp/photo.tiff")
    assert ts is None
    assert desc is None
    assert kw is None


# --- print_result ---

def test_fmt_ts_shows_both_formats():
    result = fmt_ts("1987:06:15 12:00:00")
    assert "1987:06:15 12:00:00" in result
    assert "June 15, 1987 12:00pm" in result

def test_fmt_ts_no_date():
    assert fmt_ts(None) == "[No Date]"


def test_print_result_shows_all_fields(capsys):
    data = {"timestamp": "1987:06:15 12:00:00", "summary": "found June 15"}
    print_result(data, verso_text="Summer holiday")
    out = capsys.readouterr().out
    assert "1987:06:15 12:00:00" in out
    assert "June 15, 1987 12:00pm" in out
    assert "Summer holiday" in out
    assert "found June 15" in out


def test_print_result_shows_no_date_when_null(capsys):
    data = {"timestamp": None, "summary": ""}
    print_result(data)
    out = capsys.readouterr().out
    assert "[No Date]" in out


def test_print_result_omits_empty_verso_and_summary(capsys):
    data = {"timestamp": "2000:01:01 12:00:00", "summary": ""}
    print_result(data, verso_text="")
    out = capsys.readouterr().out
    assert "Description" not in out
    assert "Reasoning" not in out


# --- prompt_accept ---

def test_prompt_accept_enter_returns_accept():
    with patch("builtins.input", return_value=""):
        assert prompt_accept() == "accept"


def test_prompt_accept_e_returns_edit():
    with patch("builtins.input", return_value="e"):
        assert prompt_accept() == "edit"


def test_prompt_accept_s_returns_skip():
    with patch("builtins.input", return_value="s"):
        assert prompt_accept() == "skip"


def test_prompt_accept_p_returns_escalate_when_allowed():
    with patch("builtins.input", return_value="p"):
        assert prompt_accept(allow_escalate=True) == "escalate"


def test_prompt_accept_p_reprompts_when_not_allowed():
    # 'p' should be ignored when allow_escalate=False, then '' accepts
    with patch("builtins.input", side_effect=["p", ""]):
        assert prompt_accept(allow_escalate=False) == "accept"


def test_prompt_accept_q_returns_quit():
    with patch("builtins.input", return_value="q"):
        assert prompt_accept() == "quit"


def test_prompt_accept_reprompts_on_invalid_input():
    with patch("builtins.input", side_effect=["x", "y", "e"]):
        assert prompt_accept() == "edit"


# --- prompt_edit ---

def test_prompt_edit_overrides_both_fields():
    data = {"timestamp": "1987:01:01 12:00:00", "description": "old desc", "summary": "old"}
    with patch("builtins.input", side_effect=["1987:06:15 12:00:00", "new desc"]):
        result = prompt_edit(data)
    assert result["timestamp"] == "1987:06:15 12:00:00"
    assert result["description"] == "new desc"


def test_prompt_edit_keeps_existing_when_empty_input():
    data = {"timestamp": "1987:01:01 12:00:00", "description": "old desc", "summary": "old"}
    with patch("builtins.input", side_effect=["", ""]):
        result = prompt_edit(data)
    assert result["timestamp"] == "1987:01:01 12:00:00"
    assert result["description"] == "old desc"


def test_prompt_edit_null_timestamp_stays_null_on_empty_input():
    data = {"timestamp": None, "description": "desc", "summary": ""}
    with patch("builtins.input", side_effect=["", ""]):
        result = prompt_edit(data)
    assert result["timestamp"] is None


def test_prompt_edit_preserves_summary():
    data = {"timestamp": None, "description": "", "summary": "original reasoning"}
    with patch("builtins.input", side_effect=["", ""]):
        result = prompt_edit(data)
    assert result["summary"] == "original reasoning"
