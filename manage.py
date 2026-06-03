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
    added = tagger.scan_directory(conn, args.archive, args.dir)
    conn.close()
    print(f"Registered {added} new scan(s) from {args.dir}")


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

    return parser


def main():
    args = make_parser().parse_args()

    dispatch = {
        "init": cmd_init,
        "scan-dir": cmd_scan_dir,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
