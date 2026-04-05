#!/usr/bin/env python3
import os
import glob
import sqlite3
import subprocess
import argparse
import json
import re
import time
import urllib.request
import urllib.error
from datetime import datetime
from google import genai
from google.genai import types


def init_db(db_path):
    """Initialize the SQLite database to track scan states."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            filename TEXT PRIMARY KEY,
            state TEXT,
            applied_timestamp TEXT,
            applied_description TEXT
        )
    ''')
    conn.commit()
    return conn


def sync_directory_to_db(conn, directory):
    """Finds all TIFFs (case-insensitive) in the directory and adds missing ones to the DB as PENDING."""
    cursor = conn.cursor()
    files = set(
        glob.glob(os.path.join(directory, "*.tiff")) +
        glob.glob(os.path.join(directory, "*.TIFF"))
    )

    new_files_count = 0
    for f in files:
        filename = os.path.basename(f)
        cursor.execute("SELECT state FROM scans WHERE filename = ?", (filename,))
        if not cursor.fetchone():
            cursor.execute("INSERT INTO scans (filename, state) VALUES (?, 'PENDING')", (filename,))
            new_files_count += 1

    conn.commit()
    if new_files_count > 0:
        print(f"Added {new_files_count} new scans to the tracking database.")


def is_file_stable(path, min_age=2.0):
    """Returns True if the file exists, is non-empty, and hasn't been modified in min_age seconds."""
    try:
        if os.path.getsize(path) == 0:
            return False
        return time.time() - os.path.getmtime(path) >= min_age
    except OSError:
        return False


def make_gemini_llm_fn(api_key):
    """Returns a callable that sends a prompt to Gemini and returns the raw response text."""
    client = genai.Client(api_key=api_key)

    def call(prompt, system_instruction):
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.1,
            )
        )
        return response.text

    return call


def make_ollama_llm_fn(model="llama3.2"):
    """Returns a callable that sends a prompt to a local Ollama instance."""

    def call(prompt, system_instruction):
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "format": "json"
        }).encode()

        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data["message"]["content"]

    return call


def check_ollama_available(model):
    """Returns True if Ollama is running and the requested model is available."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            model_names = [m["name"] for m in data.get("models", [])]
            return any(name == model or name.startswith(model + ":") for name in model_names)
    except Exception:
        return False


TIMESTAMP_RE = re.compile(r'^\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}$')


def parse_with_llm(verso_text, baseline_context, llm_fn, max_retries=3):
    """Passes text + baseline to the LLM with retry logic."""

    system_instruction = """
    You are an EXIF metadata extraction tool.
    Analyze the provided text. You will receive a 'Baseline Context' (the general event/date for the batch)
    and 'Verso Text' (specific writing on the back of this photo).

    1. Extract or infer the most likely date, prioritizing the Verso Text if it contradicts the Baseline.
    2. If only a year is found, use that year with 01 for month and day, e.g. "1987:01:01 12:00:00".
    3. If a month and year are found, use the first of that month, e.g. "1987:06:01 12:00:00".
    4. If an exact date is found, use it directly, e.g. "1987:06:15 12:00:00" (assume US MM-DD-YY format unless obvious otherwise).
    5. If no date can be inferred, return null for the timestamp.

    Respond ONLY with a JSON object containing two keys:
    - "timestamp": The calculated EXIF date string in the format "YYYY:MM:DD HH:MM:SS", or null.
    - "summary": A brief explanation of how you derived the timestamp.
    """

    prompt = f"Baseline Context: {baseline_context}\nVerso Text: {verso_text}"

    for attempt in range(max_retries):
        try:
            response_text = llm_fn(prompt, system_instruction)
            data = json.loads(response_text)
            ts = data.get('timestamp')
            if ts is not None and not TIMESTAMP_RE.match(str(ts)):
                raise ValueError(f"Invalid timestamp format returned: {ts!r}")
            return data

        except Exception as e:
            error_msg = str(e)
            if '429' in error_msg or 'RESOURCE_EXHAUSTED' in error_msg:
                if attempt < max_retries - 1:
                    match = re.search(r'retry in ([\d\.]+)s', error_msg)
                    wait_time = float(match.group(1)) + 1.0 if match else 35.0
                    print(f"\n[API Rate Limit]: Pausing for {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
            if attempt < max_retries - 1 and isinstance(e, ValueError) and 'Invalid timestamp' in str(e):
                print(f"\n[Bad response, retrying...]: {e}")
                continue
            raise e


def format_timestamp(exif_ts):
    """Converts EXIF timestamp 'YYYY:MM:DD HH:MM:SS' to 'Month D, YYYY H:MMam/pm'."""
    if not exif_ts:
        return None
    try:
        dt = datetime.strptime(exif_ts, "%Y:%m:%d %H:%M:%S")
        hour = dt.hour % 12 or 12
        ampm = "am" if dt.hour < 12 else "pm"
        return f"{dt.strftime('%B')} {dt.day}, {dt.year} {hour}:{dt.strftime('%M')}{ampm}"
    except ValueError:
        return exif_ts


def read_exif(file_path):
    """Returns (DateTimeOriginal, Description, Keywords) currently written on a file."""
    result = subprocess.run(
        ["exiftool", "-json",
         "-DateTimeOriginal", "-IPTC:DateCreated", "-XMP:CreateDate",
         "-Description", "-IPTC:Keywords", file_path],
        capture_output=True, text=True
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None, None, None
    record = json.loads(result.stdout)[0]
    return record.get("DateTimeOriginal"), record.get("Description"), record.get("Keywords")


def write_exif(file_path, timestamp, description, keywords=None):
    """Writes EXIF metadata to a file using exiftool."""
    cmd = ["exiftool", "-overwrite_original"]
    if timestamp:
        cmd.append(f"-DateTimeOriginal={timestamp}")
        cmd.append(f"-IPTC:DateCreated={timestamp}")
        cmd.append(f"-XMP:CreateDate={timestamp}")
    if description:
        cmd.append(f"-Description={description}")
    if keywords:
        cmd.append(f"-IPTC:Keywords={keywords}")
    cmd.append(file_path)
    subprocess.run(cmd, stdout=subprocess.DEVNULL)


def fmt_ts(exif_ts):
    """Returns 'YYYY:MM:DD HH:MM:SS (Month D, YYYY H:MMam/pm)' or '[No Date]'."""
    if not exif_ts:
        return "[No Date]"
    human = format_timestamp(exif_ts)
    return f"{exif_ts} ({human})" if human else exif_ts


def print_result(data, verso_text=""):
    """Prints the LLM result in a readable format."""
    print(f"   Timestamp  : {fmt_ts(data.get('timestamp'))}")
    if verso_text:
        print(f"   Description: {verso_text}")
    if data.get('summary'):
        print(f"   Reasoning  : {data['summary']}")


def prompt_accept(allow_escalate=False):
    """Prompts the user to accept, escalate, edit, skip, or quit.
    Returns one of: 'accept', 'escalate', 'edit', 'skip', 'quit'.
    """
    parts = ["[Enter] accept"]
    if allow_escalate:
        parts.append("[p] retry with Gemini")
    parts.extend(["[e] edit", "[s] skip", "[q] quit"])
    prompt = " | ".join(parts) + ": "

    while True:
        choice = input(prompt).strip().lower()
        if choice == '':
            return 'accept'
        if choice == 'p' and allow_escalate:
            return 'escalate'
        if choice == 'e':
            return 'edit'
        if choice == 's':
            return 'skip'
        if choice == 'q':
            return 'quit'


def prompt_edit(data):
    """Lets the user manually edit timestamp and description. Returns updated data dict."""
    current_ts = data.get('timestamp') or ''
    current_desc = data.get('description') or ''
    new_ts = input(f"Timestamp [{current_ts}]: ").strip()
    new_desc = input(f"Description [{current_desc}]: ").strip()
    return {
        'timestamp': new_ts if new_ts else (current_ts or None),
        'description': new_desc if new_desc else current_desc,
        'summary': data.get('summary', '')
    }


def main():
    parser = argparse.ArgumentParser(description="Shoebox & Database metadata pipeline.")
    parser.add_argument("-d", "--dir", default=".", help="Directory containing the scans")
    parser.add_argument("-m", "--local-model", default="llama3.2",
                        help="Ollama model for local inference (default: llama3.2)")
    parser.add_argument("--no-local", action="store_true", help="Skip local LLM, use Gemini directly")
    parser.add_argument("--reset", action="store_true", help="Delete the tracking database and start fresh")
    args = parser.parse_args()

    # Setup DB
    db_path = os.path.join(args.dir, ".scans.db")
    if args.reset and os.path.exists(db_path):
        os.remove(db_path)
        print("Tracking database reset.")
    conn = init_db(db_path)
    cursor = conn.cursor()

    # Determine available LLMs
    local_llm_fn = None
    paid_llm_fn = None

    if not args.no_local:
        print(f"\nChecking for local Ollama model '{args.local_model}'...")
        if check_ollama_available(args.local_model):
            local_llm_fn = make_ollama_llm_fn(args.local_model)
            print("-> Local model ready.")
        else:
            print(f"-> Ollama not available or model not found. Will use Gemini.")

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        paid_llm_fn = make_gemini_llm_fn(api_key)

    if local_llm_fn is None and paid_llm_fn is None:
        print("Error: No local LLM available and GEMINI_API_KEY not set.")
        conn.close()
        return

    # Initial sync and pending file count
    sync_directory_to_db(conn, args.dir)
    cursor.execute("SELECT COUNT(*) FROM scans WHERE state IN ('PENDING', 'MAYBE')")
    pending_count = cursor.fetchone()[0]
    if pending_count > 0:
        print(f"\nFound {pending_count} file(s) pending.")

    # Shoebox Baseline
    print("\n--- Shoebox Baseline ---")
    baseline = input("Enter baseline context for this batch (or press Enter if none): ").strip()

    first_llm = local_llm_fn if local_llm_fn else paid_llm_fn
    use_local = local_llm_fn is not None

    baseline_data = None
    if baseline:
        print(f"\nChecking baseline with {'local model' if use_local else 'Gemini'}...")
        try:
            baseline_data = parse_with_llm("", baseline, first_llm)
            print(f"[Baseline check]")
            print(f"   Default timestamp: {fmt_ts(baseline_data.get('timestamp'))}")
            print(f"   Reasoning        : {baseline_data.get('summary', '')}")
            print(f"   Keywords         : {baseline}")
            confirm = input("Continue with this baseline? [Enter] yes / [e] edit: ").strip().lower()
            if confirm == 'e':
                baseline = input("Enter revised baseline: ").strip()
                baseline_data = None  # will re-derive with verso text per photo
        except Exception as e:
            print(f"-> Baseline check failed: {e}")
    elif local_llm_fn:
        print("\nWarming up local model...", end="", flush=True)
        try:
            local_llm_fn("warmup", "Respond with valid JSON: {}")
        except Exception:
            pass
        print(" ready.")

    print("\n[Watcher Mode Active]: Listening for new scans... (Type 'q' or press Ctrl+C to stop)")

    last_preview_file = None

    try:
        while True:
            sync_directory_to_db(conn, args.dir)
            cursor.execute("SELECT filename FROM scans WHERE state IN ('PENDING', 'MAYBE') ORDER BY filename")
            queue = cursor.fetchall()

            if not queue:
                time.sleep(3)
                continue

            processed_this_round = False

            for (filename,) in queue:
                file_path = os.path.join(args.dir, filename)

                if not is_file_stable(file_path):
                    print(f"-> {filename}: still being written, will retry.")
                    continue

                print(f"\n--- Processing: {filename} ---")
                current_ts, current_desc, current_kw = read_exif(file_path)
                print(f"   Current date : {fmt_ts(current_ts) if current_ts else '[none]'}")
                print(f"   Current desc : {current_desc or '[none]'}")
                print(f"   Current keywords: {current_kw or '[none]'}")

                # Close previous Preview window before opening the next
                if last_preview_file:
                    subprocess.run(
                        ["osascript", "-e", 'tell application "Preview" to close first window'],
                        stderr=subprocess.DEVNULL
                    )
                subprocess.run(["open", "-g", file_path])
                last_preview_file = file_path

                print("Commands: [text] verso text | [Enter] baseline only | [m] maybe | [s] skip | [q] quit")
                user_input = input("Verso text: ").strip()

                if user_input.lower() == 'q':
                    if last_preview_file:
                        subprocess.run(
                            ["osascript", "-e", 'tell application "Preview" to close first window'],
                            stderr=subprocess.DEVNULL
                        )
                    print("Exiting. Progress saved.")
                    conn.close()
                    return
                elif user_input.lower() == 'm':
                    cursor.execute("UPDATE scans SET state = 'MAYBE' WHERE filename = ?", (filename,))
                    conn.commit()
                    print("-> Marked as MAYBE.")
                    continue
                elif user_input.lower() == 's':
                    cursor.execute("UPDATE scans SET state = 'SKIPPED' WHERE filename = ?", (filename,))
                    conn.commit()
                    print("-> Marked as SKIPPED.")
                    continue

                if not user_input and not baseline:
                    print("-> No baseline and no text. Skipping.")
                    cursor.execute("UPDATE scans SET state = 'SKIPPED' WHERE filename = ?", (filename,))
                    conn.commit()
                    continue

                if not user_input and baseline_data:
                    print("\n[Baseline result]")
                    data = baseline_data
                    print_result(data, user_input)
                else:
                    print("Parsing with AI...")
                    data = parse_with_llm(user_input, baseline, first_llm)
                    print(f"\n[{'Local' if use_local else 'Gemini'} result]")
                    print_result(data, user_input)

                try:
                    action = prompt_accept(allow_escalate=(use_local and paid_llm_fn is not None))

                    if action == 'escalate':
                        print("\nRetrying with Gemini...")
                        data = parse_with_llm(user_input, baseline, paid_llm_fn)
                        print("\n[Gemini result]")
                        print_result(data, user_input)
                        action = prompt_accept(allow_escalate=False)

                    if action == 'quit':
                        subprocess.run(
                            ["osascript", "-e", 'tell application "Preview" to close first window'],
                            stderr=subprocess.DEVNULL
                        )
                        print("Exiting. Progress saved.")
                        conn.close()
                        return

                    if action == 'edit':
                        data = prompt_edit(data)
                        action = 'accept'

                    if action == 'skip':
                        cursor.execute("UPDATE scans SET state = 'SKIPPED' WHERE filename = ?", (filename,))
                        conn.commit()
                        print("-> Skipped.")
                        continue

                    # accept
                    timestamp = data.get('timestamp')
                    description = user_input
                    print(f"-> Writing:")
                    print(f"   Timestamp  : {fmt_ts(timestamp)}")
                    print(f"   Description: {description or '[none]'}")
                    print(f"   Keywords   : {baseline or '[none]'}")
                    write_exif(file_path, timestamp, description, keywords=baseline or None)

                    # Verify what was written
                    v_ts, v_desc, v_kw = read_exif(file_path)
                    print(f"-> Verified:")
                    print(f"   Timestamp  : {fmt_ts(v_ts)}")
                    print(f"   Description: {v_desc or '[none]'}")
                    print(f"   Keywords   : {v_kw or '[none]'}")
                    if timestamp and not v_ts:
                        print("   WARNING: DateTimeOriginal not confirmed by exiftool")

                    cursor.execute(
                        "UPDATE scans SET state = 'PROCESSED', applied_timestamp = ?, applied_description = ? WHERE filename = ?",
                        (timestamp, description, filename)
                    )
                    conn.commit()
                    processed_this_round = True

                    if not use_local:
                        time.sleep(4)  # Gemini free tier throttle

                except Exception as e:
                    print(f"-> Failed: {e}")
                    cursor.execute("UPDATE scans SET state = 'FAILED' WHERE filename = ?", (filename,))
                    conn.commit()

            # After draining the queue, check if we're all done
            cursor.execute("SELECT COUNT(*) FROM scans WHERE state IN ('PENDING', 'MAYBE')")
            remaining = cursor.fetchone()[0]
            if processed_this_round and remaining == 0:
                print("\n--- All files processed. ---")
                choice = input("Keep watching for new files [Enter] / quit [q]: ").strip().lower()
                if choice == 'q':
                    conn.close()
                    return

    except KeyboardInterrupt:
        if last_preview_file:
            subprocess.run(
                ["osascript", "-e", 'tell application "Preview" to close first window'],
                stderr=subprocess.DEVNULL
            )
        print("\nExiting watcher. Progress saved.")
        conn.close()


if __name__ == "__main__":
    main()
