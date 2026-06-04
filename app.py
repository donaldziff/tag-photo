#!/usr/bin/env python3
import argparse
import os

from flask import Flask, g, redirect, render_template, request, send_file, url_for

import tagger

app = Flask(__name__)


# ---------------------------------------------------------------------------
# DB per request
# ---------------------------------------------------------------------------

def get_db():
    if "db" not in g:
        g.db = tagger.init_db(app.config["ARCHIVE_ROOT"])
    return g.db


@app.teardown_appcontext
def close_db(error):
    db = g.pop("db", None)
    if db is not None:
        db.close()


@app.before_request
def enforce_safe_mode():
    if app.config.get("SAFE_MODE") and request.method == "POST":
        return "Safe mode is active — all writes are disabled.", 403


@app.context_processor
def inject_globals():
    return {"safe_mode": app.config.get("SAFE_MODE", False)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pending_count(db, scan_dir):
    return db.execute(
        "SELECT COUNT(*) FROM scans WHERE scan_dir=? AND is_verso=0 AND state='PENDING'",
        (scan_dir,),
    ).fetchone()[0]


def _nav(db, scan_dir, hash_val, pending_only):
    """Return (prev_hash, next_hash) for the given photo within its sequence."""
    scans = tagger.get_scans_for_dir(db, scan_dir, pending_only=pending_only)
    hashes = [s["hash"] for s in scans]
    try:
        idx = hashes.index(hash_val)
    except ValueError:
        return None, None
    prev_hash = hashes[idx - 1] if idx > 0 else None
    next_hash = hashes[idx + 1] if idx < len(hashes) - 1 else None
    return prev_hash, next_hash


def _detail_context(db, scan_dir, hash_val, mode):
    scan = tagger.get_scan(db, hash_val)
    envelope = tagger.get_envelope(db, scan["envelope_id"])
    verso = tagger.get_scan(db, scan["verso_hash"]) if scan["verso_hash"] else None
    envelopes = tagger.list_envelopes(db)
    file_path = os.path.join(app.config["ARCHIVE_ROOT"], scan["scan_dir"], scan["filename"])
    exif = tagger.read_exif(file_path)
    pending = _pending_count(db, scan_dir)
    prev_hash, next_hash = _nav(db, scan_dir, hash_val, pending_only=(mode == "tag"))
    return dict(scan=scan, envelope=envelope, verso=verso, envelopes=envelopes,
                exif=exif, scan_dir=scan_dir, mode=mode,
                pending_count=pending, prev_hash=prev_hash, next_hash=next_hash)


# ---------------------------------------------------------------------------
# Image serving
# ---------------------------------------------------------------------------

@app.route("/thumb/<hash_val>")
def thumbnail(hash_val):
    db = get_db()
    scan = tagger.get_scan(db, hash_val)
    path = tagger.ensure_thumbnail(
        app.config["ARCHIVE_ROOT"], hash_val, scan["scan_dir"], scan["filename"]
    )
    return send_file(path, mimetype="image/jpeg")


@app.route("/image/<hash_val>")
def image(hash_val):
    db = get_db()
    scan = tagger.get_scan(db, hash_val)
    path = tagger.ensure_preview(
        app.config["ARCHIVE_ROOT"], hash_val, scan["scan_dir"], scan["filename"]
    )
    return send_file(path, mimetype="image/jpeg")


# ---------------------------------------------------------------------------
# Home — scan directory picker
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    db = get_db()
    scan_dirs = tagger.get_scan_dirs(db)
    if len(scan_dirs) == 1:
        return redirect(url_for("tag_grid", scan_dir=scan_dirs[0]))
    return render_template("index.html", scan_dirs=scan_dirs)


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

@app.route("/<scan_dir>/")
def tag_grid(scan_dir):
    db = get_db()
    scans = tagger.get_scans_for_dir(db, scan_dir, pending_only=True)
    return render_template("grid.html", scans=scans, scan_dir=scan_dir,
                           mode="tag", pending_count=len(scans))


@app.route("/<scan_dir>/browse")
def browse_grid(scan_dir):
    db = get_db()
    scans = tagger.get_scans_for_dir(db, scan_dir)
    pending = sum(1 for s in scans if s["state"] == "PENDING")
    return render_template("grid.html", scans=scans, scan_dir=scan_dir,
                           mode="browse", pending_count=pending)


# ---------------------------------------------------------------------------
# Detail pages
# ---------------------------------------------------------------------------

@app.route("/<scan_dir>/tag/<hash_val>")
def tag_detail(scan_dir, hash_val):
    db = get_db()
    return render_template("detail.html", **_detail_context(db, scan_dir, hash_val, "tag"))


@app.route("/<scan_dir>/browse/<hash_val>")
def browse_detail(scan_dir, hash_val):
    db = get_db()
    return render_template("detail.html", **_detail_context(db, scan_dir, hash_val, "browse"))


# ---------------------------------------------------------------------------
# Tag-mode actions
# ---------------------------------------------------------------------------

def _accept_form(db, hash_val):
    return dict(
        description=request.form.get("description") or None,
        verso_text=request.form.get("verso_text") or None,
        recto_stamp_text=request.form.get("recto_stamp_text") or None,
        date_inferred=request.form.get("date_inferred") or None,
        date_source=request.form.get("date_source") or None,
        envelope_id=request.form.get("envelope_id") or None,
    )


@app.route("/<scan_dir>/tag/<hash_val>/accept", methods=["POST"])
def tag_accept(scan_dir, hash_val):
    db = get_db()
    # Capture next BEFORE state change removes current from pending list
    _, next_hash = _nav(db, scan_dir, hash_val, pending_only=True)
    tagger.accept_photo(db, hash_val, app.config["ARCHIVE_ROOT"], **_accept_form(db, hash_val))
    if next_hash:
        return redirect(url_for("tag_detail", scan_dir=scan_dir, hash_val=next_hash))
    return redirect(url_for("tag_grid", scan_dir=scan_dir))


@app.route("/<scan_dir>/tag/<hash_val>/skip", methods=["POST"])
def tag_skip(scan_dir, hash_val):
    db = get_db()
    _, next_hash = _nav(db, scan_dir, hash_val, pending_only=True)
    tagger.set_scan_state(db, hash_val, "SKIPPED")
    if next_hash:
        return redirect(url_for("tag_detail", scan_dir=scan_dir, hash_val=next_hash))
    return redirect(url_for("tag_grid", scan_dir=scan_dir))


@app.route("/<scan_dir>/tag/<hash_val>/needs-pairing", methods=["POST"])
def tag_needs_pairing(scan_dir, hash_val):
    db = get_db()
    _, next_hash = _nav(db, scan_dir, hash_val, pending_only=True)
    tagger.set_scan_state(db, hash_val, "NEEDS_PAIRING")
    if next_hash:
        return redirect(url_for("tag_detail", scan_dir=scan_dir, hash_val=next_hash))
    return redirect(url_for("tag_grid", scan_dir=scan_dir))


# ---------------------------------------------------------------------------
# Browse-mode actions
# ---------------------------------------------------------------------------

@app.route("/<scan_dir>/browse/<hash_val>/accept", methods=["POST"])
def browse_accept(scan_dir, hash_val):
    db = get_db()
    tagger.accept_photo(db, hash_val, app.config["ARCHIVE_ROOT"], **_accept_form(db, hash_val))
    return redirect(url_for("browse_detail", scan_dir=scan_dir, hash_val=hash_val))


@app.route("/<scan_dir>/browse/<hash_val>/edit", methods=["POST"])
def browse_edit(scan_dir, hash_val):
    db = get_db()
    tagger.reopen_photo(db, hash_val)
    return redirect(url_for("browse_detail", scan_dir=scan_dir, hash_val=hash_val))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Photo tagging web UI")
    parser.add_argument("-a", "--archive", required=True, metavar="ARCHIVE_ROOT")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--safe", action="store_true",
                        help="Read-only mode: disable all writes")
    args = parser.parse_args()
    app.config["ARCHIVE_ROOT"] = os.path.expanduser(args.archive)
    app.config["SAFE_MODE"] = args.safe
    if args.safe:
        print("Safe mode: all writes disabled.")
    app.run(debug=True, port=args.port)


if __name__ == "__main__":
    main()
