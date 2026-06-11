#!/usr/bin/env python3
import argparse
import os
import sys

import tagger


# --- subcommand handlers ---

def cmd_init(args):
    os.makedirs(args.archive, exist_ok=True)
    conn = tagger.init_db(args.archive)
    conn.close()
    print(f"Initialized archive at {args.archive}")


def cmd_scan_dir(args):
    scan_dir_path = os.path.join(args.archive, args.dir)
    if not os.path.isdir(scan_dir_path):
        print(f"Error: scan directory not found: {scan_dir_path}", file=sys.stderr)
        sys.exit(1)
    conn = tagger.init_db(args.archive)
    if args.envelope:
        result = tagger.upsert_envelope(conn, args.envelope, args.envelope_desc)
        if result == "updated":
            print(f"Updated envelope {args.envelope} description to: {args.envelope_desc}")
        elif result == "created" and args.envelope_desc:
            print(f"Created envelope {args.envelope}: {args.envelope_desc}")
    added = tagger.scan_directory(conn, args.archive, args.dir, envelope_id=args.envelope)
    conn.close()
    envelope_note = f" (envelope {args.envelope})" if args.envelope else ""
    print(f"Registered {added} new scan(s) from {args.dir}{envelope_note}")


def cmd_update_envelope(args):
    conn = tagger.init_db(args.archive)
    result = tagger.upsert_envelope(conn, args.envelope, args.desc)
    conn.close()
    if result == "created":
        print(f"Created envelope {args.envelope}: {args.desc}")
    elif result == "updated":
        print(f"Updated envelope {args.envelope}: {args.desc}")
    else:
        print(f"Envelope {args.envelope} description unchanged.")


def cmd_list_envelopes(args):
    conn = tagger.init_db(args.archive)
    rows = tagger.list_envelopes(conn)
    conn.close()
    if not rows:
        print("No envelopes found.")
        return
    for envelope_id, description, scan_count in rows:
        desc_str = description or "(no description)"
        print(f"  {envelope_id:>6}  {scan_count:>3} scan(s)  {desc_str}")


def cmd_clear_thumbs(args):
    thumbs_dir = os.path.join(args.archive, ".thumbs")
    if not os.path.isdir(thumbs_dir):
        print("No thumbnail cache found.")
        return
    files = [f for f in os.listdir(thumbs_dir) if f.endswith(".jpg")]
    for f in files:
        os.remove(os.path.join(thumbs_dir, f))
    print(f"Cleared {len(files)} cached thumbnail/preview file(s).")


def cmd_prune_dir(args):
    import tagger as _tagger
    conn = _tagger.init_db(args.archive)
    dir_path = os.path.join(args.archive, args.dir)

    rows = conn.execute(
        "SELECT filename, COUNT(*) n FROM scans WHERE scan_dir=? GROUP BY filename HAVING n > 1",
        (args.dir,),
    ).fetchall()

    if not rows:
        print("No duplicate filenames found.")
        conn.close()
        return

    fixed = 0
    for (filename, _) in rows:
        file_path = os.path.join(dir_path, filename)
        if not os.path.exists(file_path):
            print(f"  SKIP {filename}: file not on disk")
            continue

        live_hash = _tagger.hash_file(file_path)
        records = conn.execute(
            "SELECT * FROM scans WHERE scan_dir=? AND filename=?", (args.dir, filename)
        ).fetchall()

        live = next((r for r in records if r["hash"] == live_hash), None)
        stale = next((r for r in records if r["hash"] != live_hash), None)

        if not live or not stale:
            print(f"  SKIP {filename}: couldn't identify live/stale records")
            continue

        # Transfer non-null metadata from stale to live if live is missing it
        fields = ("verso_hash", "envelope_id", "description", "verso_text",
                  "recto_stamp_text", "date_inferred", "date_source",
                  "jpeg_path", "uploaded_at")
        updates = {f: stale[f] for f in fields if stale[f] is not None and live[f] is None}
        # Prefer stale state if it carries more meaning than PENDING
        if stale["state"] != "PENDING" and live["state"] == "PENDING":
            updates["state"] = stale["state"]

        if updates:
            set_clause = ", ".join(f"{k}=?" for k in updates)
            conn.execute(
                f"UPDATE scans SET {set_clause} WHERE hash=?",
                (*updates.values(), live_hash),
            )

        # If any scan points its verso_hash at the stale record, redirect it to live
        conn.execute(
            "UPDATE scans SET verso_hash=? WHERE verso_hash=?", (live_hash, stale["hash"])
        )

        conn.execute("DELETE FROM scans WHERE hash=?", (stale["hash"],))
        print(f"  {filename}: merged stale {stale['hash'][:8]}… → live {live_hash[:8]}…")
        fixed += 1

    conn.commit()
    conn.close()
    print(f"Fixed {fixed} duplicate(s).")


def cmd_export(args):
    scan_dir_path = os.path.join(args.archive, args.dir)
    if not os.path.isdir(scan_dir_path):
        print(f"Error: scan directory not found: {scan_dir_path}", file=sys.stderr)
        sys.exit(1)
    conn = tagger.init_db(args.archive)
    exported = tagger.export_directory(conn, args.archive, args.dir)
    conn.close()
    print(f"Exported {exported} JPEG(s) to {os.path.join(args.dir, 'export')}/")


def cmd_mark_uploaded(args):
    conn = tagger.init_db(args.archive)
    count = tagger.mark_uploaded(conn, args.dir)
    conn.close()
    print(f"Marked {count} scan(s) as uploaded in {args.dir}")


def cmd_migrate_dir(args):
    scan_dir_path = os.path.join(args.archive, args.dir)
    if not os.path.isdir(scan_dir_path):
        print(f"Error: scan directory not found: {scan_dir_path}", file=sys.stderr)
        sys.exit(1)
    conn = tagger.init_db(args.archive)
    added, skipped = tagger.migrate_directory(conn, args.archive, args.dir)
    conn.close()
    print(f"{args.dir}: {added} imported, {skipped} already in DB")


def cmd_mark_verso(args):
    conn = tagger.init_db(args.archive)
    pair = tagger.get_recent_pair(conn, args.dir)
    if pair is None:
        print("Error: need at least 2 scans in this directory.", file=sys.stderr)
        conn.close()
        sys.exit(1)
    (recto_hash, recto_filename), (verso_hash, verso_filename) = pair
    print(f"  Recto : {recto_filename}")
    print(f"  Verso : {verso_filename}")
    confirm = input("Mark as recto/verso pair? [y/N]: ").strip().lower()
    if confirm == "y":
        tagger.set_verso_pair(conn, recto_hash, verso_hash)
        print("-> Paired.")
    else:
        print("-> Cancelled.")
    conn.close()


# --- argument parsing ---

def make_parser():
    # Common -a flag inherited by all subcommands
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-a", "--archive", required=True, metavar="ARCHIVE_ROOT",
                        help="Archive root directory")

    parser = argparse.ArgumentParser(
        description="Photo archive management tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", parents=[common],
                   help="Initialize a new archive (creates tagger.db)")

    p_scan = sub.add_parser("scan-dir", parents=[common],
                             help="Hash TIFFs in a scan directory and register new ones")
    p_scan.add_argument("-d", "--dir", required=True, metavar="SCAN_DIR",
                        help="Scan subdirectory name (relative to archive root)")
    p_scan.add_argument("-e", "--envelope", metavar="ENVELOPE_ID",
                        help="Envelope ID to assign to newly registered scans")
    p_scan.add_argument("--envelope-desc", metavar="DESCRIPTION",
                        help="Envelope description (always updated if provided)")

    p_env = sub.add_parser("update-envelope", parents=[common],
                            help="Set or correct an envelope description")
    p_env.add_argument("-e", "--envelope", required=True, metavar="ENVELOPE_ID",
                       help="Envelope ID")
    p_env.add_argument("--desc", required=True, metavar="DESCRIPTION",
                       help="New description")

    sub.add_parser("list-envelopes", parents=[common],
                   help="List all envelopes with scan counts")

    sub.add_parser("clear-thumbs", parents=[common],
                   help="Delete all cached thumbnails and previews (regenerated on next view)")

    p_prune = sub.add_parser("prune-dir", parents=[common],
                              help="Fix duplicate filename records caused by file edits")
    p_prune.add_argument("-d", "--dir", required=True, metavar="SCAN_DIR")

    p_export = sub.add_parser("export", parents=[common],
                              help="Export REVIEWED photos to JPEG in <scan_dir>/export/")
    p_export.add_argument("-d", "--dir", required=True, metavar="SCAN_DIR")

    p_upload = sub.add_parser("mark-uploaded", parents=[common],
                               help="Mark all EXPORTED scans in a scan dir as UPLOADED")
    p_upload.add_argument("-d", "--dir", required=True, metavar="SCAN_DIR")

    p_migrate = sub.add_parser("migrate-dir", parents=[common],
                               help="Import a scan directory from the old tag_photo.py workflow")
    p_migrate.add_argument("-d", "--dir", required=True, metavar="SCAN_DIR",
                           help="Scan subdirectory name (relative to archive root)")

    p_verso = sub.add_parser("mark-verso", parents=[common],
                              help="Pair the two most recently added scans as recto/verso")
    p_verso.add_argument("-d", "--dir", required=True, metavar="SCAN_DIR",
                         help="Scan subdirectory name (relative to archive root)")

    return parser


def main():
    args = make_parser().parse_args()

    dispatch = {
        "init": cmd_init,
        "scan-dir": cmd_scan_dir,
        "update-envelope": cmd_update_envelope,
        "list-envelopes": cmd_list_envelopes,
        "clear-thumbs": cmd_clear_thumbs,
        "prune-dir": cmd_prune_dir,
        "export": cmd_export,
        "mark-uploaded": cmd_mark_uploaded,
        "migrate-dir": cmd_migrate_dir,
        "mark-verso": cmd_mark_verso,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
