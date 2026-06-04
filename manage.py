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
        "mark-verso": cmd_mark_verso,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
