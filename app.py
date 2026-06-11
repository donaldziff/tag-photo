#!/usr/bin/env python3
import argparse
import os

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, abort, g, redirect, render_template, request, send_file, url_for

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
    db = get_db()
    return {
        "safe_mode": app.config.get("SAFE_MODE", False),
        "has_llm": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "scan_dirs": tagger.get_scan_dirs(db),
        "archive_name": os.path.basename(app.config["ARCHIVE_ROOT"]),
        "all_envelopes": tagger.list_envelopes(db),
        "llm_usage": tagger.get_llm_usage_summary(db),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pending_count(db, scan_dir):
    return db.execute(
        "SELECT COUNT(*) FROM scans WHERE scan_dir=? AND is_verso=0 AND state='PENDING'",
        (scan_dir,),
    ).fetchone()[0]


def _nav(db, scan_dir, hash_val, pending_only):
    scans = tagger.get_scans_for_dir(db, scan_dir, pending_only=pending_only)
    hashes = [s["hash"] for s in scans]
    try:
        idx = hashes.index(hash_val)
    except ValueError:
        return None, None
    return (hashes[idx - 1] if idx > 0 else None,
            hashes[idx + 1] if idx < len(hashes) - 1 else None)


def _envelope_nav(db, envelope_id, hash_val, pending_only):
    scans = tagger.get_scans_for_envelope(db, envelope_id, pending_only=pending_only)
    hashes = [s["hash"] for s in scans]
    try:
        idx = hashes.index(hash_val)
    except ValueError:
        return None, None
    return (hashes[idx - 1] if idx > 0 else None,
            hashes[idx + 1] if idx < len(hashes) - 1 else None)


def _detail_context(db, scan_dir, hash_val, mode):
    scan = tagger.get_scan(db, hash_val)
    envelope = tagger.get_envelope(db, scan["envelope_id"])
    verso = tagger.get_scan(db, scan["verso_hash"]) if scan["verso_hash"] else None
    envelopes = tagger.list_envelopes(db)
    file_path = os.path.join(app.config["ARCHIVE_ROOT"], scan["scan_dir"], scan["filename"])
    exif = tagger.read_exif(file_path)
    pending = _pending_count(db, scan_dir)
    prev_hash, next_hash = _nav(db, scan_dir, hash_val, pending_only=(mode == "tag"))

    scope = {"type": "scan_dir", "scan_dir": scan_dir, "label": scan_dir}
    det = "tag_detail" if mode == "tag" else "browse_detail"
    prev_url = url_for(det, scan_dir=scan_dir, hash_val=prev_hash) if prev_hash else None
    next_url = url_for(det, scan_dir=scan_dir, hash_val=next_hash) if next_hash else None
    grid_url = url_for("tag_grid" if mode == "tag" else "browse_grid", scan_dir=scan_dir)
    urls = {
        "accept":         url_for("tag_accept" if mode == "tag" else "browse_accept", scan_dir=scan_dir, hash_val=hash_val),
        "needs_pairing":  url_for("tag_needs_pairing", scan_dir=scan_dir, hash_val=hash_val),
        "browse_edit":    url_for("browse_edit", scan_dir=scan_dir, hash_val=hash_val),
        "delete":         url_for("delete_scans", scan_dir=scan_dir),
        "delete_next":    next_url or prev_url or grid_url,
        "extract_verso":  url_for("extract_verso", scan_dir=scan_dir, hash_val=hash_val),
        "infer_date":     url_for("infer_date", scan_dir=scan_dir, hash_val=hash_val),
        "rotate":         url_for("rotate_scan", scan_dir=scan_dir, hash_val=hash_val),
        "regen_thumb":    url_for("regen_thumb", scan_dir=scan_dir, hash_val=hash_val),
        "swap_verso":     url_for("swap_verso", scan_dir=scan_dir, hash_val=hash_val),
        "swap_verso_next": url_for("browse_detail", scan_dir=scan_dir, hash_val=scan["verso_hash"]) if scan["verso_hash"] else "",
        "prev":           prev_url,
        "next":           next_url,
        "tag_grid":       url_for("tag_grid", scan_dir=scan_dir),
        "browse_grid":    url_for("browse_grid", scan_dir=scan_dir),
    }
    return dict(scan=scan, envelope=envelope, verso=verso, envelopes=envelopes,
                exif=exif, scan_dir=scan_dir, mode=mode, scope=scope,
                pending_count=pending, prev_hash=prev_hash, next_hash=next_hash,
                urls=urls)


def _envelope_detail_context(db, envelope_id, hash_val, mode):
    scan = tagger.get_scan(db, hash_val)
    envelope = tagger.get_envelope(db, scan["envelope_id"])
    verso = tagger.get_scan(db, scan["verso_hash"]) if scan["verso_hash"] else None
    envelopes = tagger.list_envelopes(db)
    scan_dir = scan["scan_dir"]
    file_path = os.path.join(app.config["ARCHIVE_ROOT"], scan_dir, scan["filename"])
    exif = tagger.read_exif(file_path)
    all_scans = tagger.get_scans_for_envelope(db, envelope_id)
    pending = sum(1 for s in all_scans if s["state"] == "PENDING")
    prev_hash, next_hash = _envelope_nav(db, envelope_id, hash_val, pending_only=(mode == "tag"))

    scope = {"type": "envelope", "envelope_id": envelope_id, "label": f"Envelope {envelope_id}"}
    det = "envelope_tag_detail" if mode == "tag" else "envelope_browse_detail"
    prev_url = url_for(det, envelope_id=envelope_id, hash_val=prev_hash) if prev_hash else None
    next_url = url_for(det, envelope_id=envelope_id, hash_val=next_hash) if next_hash else None
    grid_url = url_for("envelope_tag_grid" if mode == "tag" else "envelope_browse_grid", envelope_id=envelope_id)
    urls = {
        "accept":         url_for("envelope_tag_accept" if mode == "tag" else "envelope_browse_accept", envelope_id=envelope_id, hash_val=hash_val),
        "needs_pairing":  url_for("envelope_tag_needs_pairing", envelope_id=envelope_id, hash_val=hash_val),
        "browse_edit":    url_for("envelope_browse_edit", envelope_id=envelope_id, hash_val=hash_val),
        "delete":         url_for("delete_scans", scan_dir=scan_dir),
        "delete_next":    next_url or prev_url or grid_url,
        "extract_verso":  url_for("extract_verso", scan_dir=scan_dir, hash_val=hash_val),
        "infer_date":     url_for("infer_date", scan_dir=scan_dir, hash_val=hash_val),
        "swap_verso":     url_for("swap_verso", scan_dir=scan_dir, hash_val=hash_val),
        "swap_verso_next": url_for("envelope_browse_detail", envelope_id=envelope_id, hash_val=scan["verso_hash"]) if scan["verso_hash"] else "",
        "prev":           prev_url,
        "next":           next_url,
        "tag_grid":       url_for("envelope_tag_grid", envelope_id=envelope_id),
        "browse_grid":    url_for("envelope_browse_grid", envelope_id=envelope_id),
    }
    return dict(scan=scan, envelope=envelope, verso=verso, envelopes=envelopes,
                exif=exif, scan_dir=scan_dir, mode=mode, scope=scope,
                pending_count=pending, prev_hash=prev_hash, next_hash=next_hash,
                urls=urls)


# ---------------------------------------------------------------------------
# Image serving
# ---------------------------------------------------------------------------

@app.route("/thumb/<hash_val>")
def thumbnail(hash_val):
    db = get_db()
    scan = tagger.get_scan(db, hash_val)
    try:
        path = tagger.ensure_thumbnail(
            app.config["ARCHIVE_ROOT"], hash_val, scan["scan_dir"], scan["filename"]
        )
    except FileNotFoundError:
        abort(404)
    return send_file(path, mimetype="image/jpeg")


@app.route("/image/<hash_val>")
def image(hash_val):
    db = get_db()
    scan = tagger.get_scan(db, hash_val)
    try:
        path = tagger.ensure_preview(
            app.config["ARCHIVE_ROOT"], hash_val, scan["scan_dir"], scan["filename"]
        )
    except FileNotFoundError:
        abort(404)
    return send_file(path, mimetype="image/jpeg")


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    db = get_db()
    scan_dirs = tagger.get_scan_dirs(db)
    envelopes = tagger.list_envelopes(db)
    if len(scan_dirs) == 1 and not envelopes:
        return redirect(url_for("tag_grid", scan_dir=scan_dirs[0]))
    return render_template("index.html", scan_dirs=scan_dirs, envelopes=envelopes)


# ---------------------------------------------------------------------------
# Envelopes management
# ---------------------------------------------------------------------------

@app.route("/envelopes")
def envelopes_index():
    db = get_db()
    envelopes = tagger.get_envelopes_with_thumbnails(db)
    unassigned = tagger.get_scans_unassigned(db)
    unassigned_pending = sum(1 for s in unassigned if s["state"] == "PENDING")
    return render_template("envelopes.html", envelopes=envelopes,
                           unassigned_count=len(unassigned),
                           unassigned_pending=unassigned_pending)


@app.route("/envelopes/new", methods=["POST"])
def envelope_new():
    db = get_db()
    env_id = request.form.get("id", "").strip()
    desc = request.form.get("description", "").strip() or None
    if env_id:
        tagger.upsert_envelope(db, env_id, desc)
    return redirect(url_for("envelopes_index"))


@app.route("/envelope/<envelope_id>/update", methods=["POST"])
def envelope_update(envelope_id):
    db = get_db()
    desc = request.form.get("description", "").strip() or None
    tagger.upsert_envelope(db, envelope_id, desc)
    return redirect(url_for("envelopes_index"))


@app.route("/assign-envelope", methods=["POST"])
def assign_envelope():
    db = get_db()
    envelope_id = request.form.get("envelope_id") or None
    for h in request.form.getlist("hash"):
        db.execute("UPDATE scans SET envelope_id=? WHERE hash=?", (envelope_id, h))
    db.commit()
    return redirect(request.form.get("next") or url_for("index"))


# ---------------------------------------------------------------------------
# Unassigned grid
# ---------------------------------------------------------------------------

@app.route("/unassigned/")
def unassigned_tag_grid():
    db = get_db()
    scans = tagger.get_scans_unassigned(db, pending_only=True)
    scope = {"type": "unassigned", "label": "No envelope assigned"}
    return render_template("grid.html", scans=scans, scope=scope,
                           mode="tag", pending_count=len(scans),
                           grid_pair_url="", grid_delete_url="")


@app.route("/unassigned/browse")
def unassigned_browse_grid():
    db = get_db()
    scans = tagger.get_scans_unassigned(db)
    pending = sum(1 for s in scans if s["state"] == "PENDING")
    scope = {"type": "unassigned", "label": "No envelope assigned"}
    return render_template("grid.html", scans=scans, scope=scope,
                           mode="browse", pending_count=pending,
                           grid_pair_url="", grid_delete_url="")


# ---------------------------------------------------------------------------
# Scan-dir grids
# ---------------------------------------------------------------------------

@app.route("/<scan_dir>/")
def tag_grid(scan_dir):
    db = get_db()
    scans = tagger.get_scans_for_dir(db, scan_dir, pending_only=True)
    scope = {"type": "scan_dir", "scan_dir": scan_dir, "label": scan_dir}
    return render_template("grid.html", scans=scans, scan_dir=scan_dir, scope=scope,
                           mode="tag", pending_count=len(scans),
                           grid_pair_url=url_for("pair_scans", scan_dir=scan_dir),
                           grid_delete_url=url_for("delete_scans", scan_dir=scan_dir))


@app.route("/<scan_dir>/browse")
def browse_grid(scan_dir):
    db = get_db()
    scans = tagger.get_scans_for_dir(db, scan_dir)
    pending = sum(1 for s in scans if s["state"] == "PENDING")
    reviewed = sum(1 for s in scans if s["state"] == "REVIEWED")
    scope = {"type": "scan_dir", "scan_dir": scan_dir, "label": scan_dir}
    return render_template("grid.html", scans=scans, scan_dir=scan_dir, scope=scope,
                           mode="browse", pending_count=pending, reviewed_count=reviewed,
                           grid_pair_url=url_for("pair_scans", scan_dir=scan_dir),
                           grid_delete_url=url_for("delete_scans", scan_dir=scan_dir))


# ---------------------------------------------------------------------------
# Envelope grids
# ---------------------------------------------------------------------------

@app.route("/envelope/<envelope_id>/")
def envelope_tag_grid(envelope_id):
    db = get_db()
    scans = tagger.get_scans_for_envelope(db, envelope_id, pending_only=True)
    scope = {"type": "envelope", "envelope_id": envelope_id, "label": f"Envelope {envelope_id}"}
    return render_template("grid.html", scans=scans, scope=scope,
                           mode="tag", pending_count=len(scans),
                           grid_pair_url=url_for("envelope_pair_scans", envelope_id=envelope_id),
                           grid_delete_url=url_for("envelope_delete_scans", envelope_id=envelope_id))


@app.route("/envelope/<envelope_id>/browse")
def envelope_browse_grid(envelope_id):
    db = get_db()
    scans = tagger.get_scans_for_envelope(db, envelope_id)
    pending = sum(1 for s in scans if s["state"] == "PENDING")
    scope = {"type": "envelope", "envelope_id": envelope_id, "label": f"Envelope {envelope_id}"}
    return render_template("grid.html", scans=scans, scope=scope,
                           mode="browse", pending_count=pending,
                           grid_pair_url=url_for("envelope_pair_scans", envelope_id=envelope_id),
                           grid_delete_url=url_for("envelope_delete_scans", envelope_id=envelope_id))


# ---------------------------------------------------------------------------
# Scan-dir detail pages
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
# Envelope detail pages
# ---------------------------------------------------------------------------

@app.route("/envelope/<envelope_id>/tag/<hash_val>")
def envelope_tag_detail(envelope_id, hash_val):
    db = get_db()
    return render_template("detail.html", **_envelope_detail_context(db, envelope_id, hash_val, "tag"))


@app.route("/envelope/<envelope_id>/browse/<hash_val>")
def envelope_browse_detail(envelope_id, hash_val):
    db = get_db()
    return render_template("detail.html", **_envelope_detail_context(db, envelope_id, hash_val, "browse"))


# ---------------------------------------------------------------------------
# Shared form helper
# ---------------------------------------------------------------------------

def _accept_form(db, hash_val):
    return dict(
        description=request.form.get("description") or None,
        verso_text=request.form.get("verso_text") or None,
        recto_stamp_text=request.form.get("recto_stamp_text") or None,
        date_inferred=request.form.get("date_inferred") or None,
        date_source=request.form.get("date_source") or None,
        date_confidence=request.form.get("date_confidence") or None,
        envelope_id=request.form.get("envelope_id") or None,
        subjects=request.form.get("subjects") or None,
    )


# ---------------------------------------------------------------------------
# Scan-dir tag actions
# ---------------------------------------------------------------------------

@app.route("/<scan_dir>/tag/<hash_val>/accept", methods=["POST"])
def tag_accept(scan_dir, hash_val):
    db = get_db()
    _, next_hash = _nav(db, scan_dir, hash_val, pending_only=True)
    tagger.accept_photo(db, hash_val, app.config["ARCHIVE_ROOT"], **_accept_form(db, hash_val))
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
# Envelope tag actions
# ---------------------------------------------------------------------------

@app.route("/envelope/<envelope_id>/tag/<hash_val>/accept", methods=["POST"])
def envelope_tag_accept(envelope_id, hash_val):
    db = get_db()
    _, next_hash = _envelope_nav(db, envelope_id, hash_val, pending_only=True)
    tagger.accept_photo(db, hash_val, app.config["ARCHIVE_ROOT"], **_accept_form(db, hash_val))
    if next_hash:
        return redirect(url_for("envelope_tag_detail", envelope_id=envelope_id, hash_val=next_hash))
    return redirect(url_for("envelope_tag_grid", envelope_id=envelope_id))


@app.route("/envelope/<envelope_id>/tag/<hash_val>/needs-pairing", methods=["POST"])
def envelope_tag_needs_pairing(envelope_id, hash_val):
    db = get_db()
    _, next_hash = _envelope_nav(db, envelope_id, hash_val, pending_only=True)
    tagger.set_scan_state(db, hash_val, "NEEDS_PAIRING")
    if next_hash:
        return redirect(url_for("envelope_tag_detail", envelope_id=envelope_id, hash_val=next_hash))
    return redirect(url_for("envelope_tag_grid", envelope_id=envelope_id))


# ---------------------------------------------------------------------------
# Scan-dir browse actions
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
# Envelope browse actions
# ---------------------------------------------------------------------------

@app.route("/envelope/<envelope_id>/browse/<hash_val>/accept", methods=["POST"])
def envelope_browse_accept(envelope_id, hash_val):
    db = get_db()
    tagger.accept_photo(db, hash_val, app.config["ARCHIVE_ROOT"], **_accept_form(db, hash_val))
    return redirect(url_for("envelope_browse_detail", envelope_id=envelope_id, hash_val=hash_val))


@app.route("/envelope/<envelope_id>/browse/<hash_val>/edit", methods=["POST"])
def envelope_browse_edit(envelope_id, hash_val):
    db = get_db()
    tagger.reopen_photo(db, hash_val)
    return redirect(url_for("envelope_browse_detail", envelope_id=envelope_id, hash_val=hash_val))


# ---------------------------------------------------------------------------
# Pairing & deletion
# ---------------------------------------------------------------------------

@app.route("/<scan_dir>/scan", methods=["POST"])
def scan_for_new(scan_dir):
    db = get_db()
    added = tagger.scan_directory(db, app.config["ARCHIVE_ROOT"], scan_dir)
    return redirect(url_for("browse_grid", scan_dir=scan_dir, added=added))


@app.route("/<scan_dir>/export", methods=["POST"])
def export_dir(scan_dir):
    db = get_db()
    count = tagger.export_directory(db, app.config["ARCHIVE_ROOT"], scan_dir)
    return redirect(url_for("browse_grid", scan_dir=scan_dir, exported=count))


@app.route("/<scan_dir>/pair", methods=["POST"])
def pair_scans(scan_dir):
    db = get_db()
    tagger.set_verso_pair(db, request.form["recto_hash"], request.form["verso_hash"])
    db.commit()
    return redirect(request.form.get("next") or url_for("browse_grid", scan_dir=scan_dir))


@app.route("/envelope/<envelope_id>/pair", methods=["POST"])
def envelope_pair_scans(envelope_id):
    db = get_db()
    tagger.set_verso_pair(db, request.form["recto_hash"], request.form["verso_hash"])
    db.commit()
    return redirect(url_for("envelope_browse_grid", envelope_id=envelope_id))


@app.route("/<scan_dir>/delete", methods=["POST"])
def delete_scans(scan_dir):
    if app.config.get("SAFE_MODE"):
        abort(403)
    db = get_db()
    for h in request.form.getlist("hash"):
        tagger.delete_scan(db, app.config["ARCHIVE_ROOT"], h)
    db.commit()
    return redirect(request.form.get("next") or url_for("browse_grid", scan_dir=scan_dir))


@app.route("/envelope/<envelope_id>/delete", methods=["POST"])
def envelope_delete_scans(envelope_id):
    if app.config.get("SAFE_MODE"):
        abort(403)
    db = get_db()
    for h in request.form.getlist("hash"):
        tagger.delete_scan(db, app.config["ARCHIVE_ROOT"], h)
    db.commit()
    return redirect(url_for("envelope_browse_grid", envelope_id=envelope_id))


# ---------------------------------------------------------------------------
# LLM + swap (use referrer / next for redirect — work in both contexts)
# ---------------------------------------------------------------------------

@app.route("/<scan_dir>/extract-verso/<hash_val>", methods=["POST"])
def extract_verso(scan_dir, hash_val):
    db = get_db()
    scan = tagger.get_scan(db, hash_val)
    verso = tagger.get_scan(db, scan["verso_hash"])
    preview_path = tagger.ensure_preview(
        app.config["ARCHIVE_ROOT"], verso["hash"], verso["scan_dir"], verso["filename"]
    )
    llm_fn = tagger.make_anthropic_llm_fn(conn=db)
    verso_text = tagger.extract_verso_text(llm_fn, preview_path)
    db.execute("UPDATE scans SET verso_text=? WHERE hash=?", (verso_text or "", hash_val))
    db.commit()
    return redirect(request.referrer or url_for("browse_detail", scan_dir=scan_dir, hash_val=hash_val))


@app.route("/<scan_dir>/infer-date/<hash_val>", methods=["POST"])
def infer_date(scan_dir, hash_val):
    db = get_db()
    scan = tagger.get_scan(db, hash_val)
    preview_path = tagger.ensure_preview(
        app.config["ARCHIVE_ROOT"], scan["hash"], scan["scan_dir"], scan["filename"]
    )
    llm_fn = tagger.make_anthropic_llm_fn(conn=db)
    result = tagger.infer_date(llm_fn, preview_path, scan["verso_text"], scan["recto_stamp_text"])
    db.execute(
        "UPDATE scans SET date_inferred=?, date_source=?, date_confidence=? WHERE hash=?",
        (result.get("date"), result.get("date_source"), result.get("date_confidence"), hash_val),
    )
    db.commit()
    return redirect(request.referrer or url_for("browse_detail", scan_dir=scan_dir, hash_val=hash_val))


@app.route("/<scan_dir>/regen-thumb/<hash_val>", methods=["POST"])
def regen_thumb(scan_dir, hash_val):
    thumbs = tagger._thumbs_dir(app.config["ARCHIVE_ROOT"])
    for suffix in ("_thumb.jpg", "_preview.jpg"):
        path = os.path.join(thumbs, f"{hash_val}{suffix}")
        if os.path.exists(path):
            os.remove(path)
    return redirect(request.referrer or url_for("browse_detail", scan_dir=scan_dir, hash_val=hash_val))


@app.route("/<scan_dir>/rotate/<hash_val>", methods=["POST"])
def rotate_scan(scan_dir, hash_val):
    degrees = -90 if request.form.get("direction") == "cw" else 90
    db = get_db()
    new_hash = tagger.rotate_scan_file(db, app.config["ARCHIVE_ROOT"], hash_val, degrees)
    next_url = request.form.get("next", "")
    if next_url and hash_val in next_url:
        return redirect(next_url.replace(hash_val, new_hash))
    return redirect(url_for("browse_detail", scan_dir=scan_dir, hash_val=new_hash))


@app.route("/<scan_dir>/swap-verso/<hash_val>", methods=["POST"])
def swap_verso(scan_dir, hash_val):
    db = get_db()
    scan = tagger.get_scan(db, hash_val)
    new_recto_hash = scan["verso_hash"]
    tagger.swap_verso_pair(db, hash_val, new_recto_hash)
    next_url = request.form.get("next") or url_for("browse_detail", scan_dir=scan_dir, hash_val=new_recto_hash)
    return redirect(next_url)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Photo tagging web UI")
    parser.add_argument("-a", "--archive", required=True, metavar="ARCHIVE_ROOT")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (use 0.0.0.0 to allow access from other devices on the LAN)")
    parser.add_argument("--safe", action="store_true",
                        help="Read-only mode: disable all writes")
    args = parser.parse_args()
    app.config["ARCHIVE_ROOT"] = os.path.expanduser(args.archive)
    app.config["SAFE_MODE"] = args.safe
    if args.safe:
        print("Safe mode: all writes disabled.")
    app.run(debug=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
