#!/usr/bin/env python3
"""
Inbox Analyzer for mmuxer

Connects to an IMAP server, analyzes unsorted Inbox messages, cross-references
against existing mmuxer rules, and interactively proposes new YAML rules.

All processing happens locally — no email data is sent anywhere.

Usage:
    python inbox_analyzer.py --config /path/to/mmuxer/config.yaml
    python inbox_analyzer.py --config /path/to/mmuxer/config.yaml --server imap.example.com --username me@example.com
    python inbox_analyzer.py --config /path/to/mmuxer/config.yaml --limit 500
"""

import argparse
import email
import email.header
import getpass
import imaplib
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file, ignoring comments and blank lines."""
    env: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip().lower()] = value.strip()
    except FileNotFoundError:
        pass
    return env


try:
    import yaml
except ImportError:
    print("PyYAML is required. Install it with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MessageInfo:
    """Minimal header info extracted from a single message."""
    uid: str
    from_addr: str  # normalized email address
    from_display: str  # display name
    to_addrs: list[str]
    subject: str
    list_id: Optional[str]
    date: str


@dataclass
class SenderGroup:
    """A cluster of messages sharing the same sender address."""
    from_addr: str
    from_display: str
    count: int
    sample_subjects: list[str]
    list_id: Optional[str] = None
    to_addrs: set = field(default_factory=set)


# ---------------------------------------------------------------------------
# mmuxer config parsing
# ---------------------------------------------------------------------------

def load_mmuxer_config(config_path: str) -> dict:
    """Load and return the mmuxer YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_covered_patterns(config: dict, ignore_folders: set[str] | None = None) -> dict:
    """
    Extract all FROM/TO/SUBJECT patterns from existing mmuxer rules.
    Returns a dict with keys 'from', 'to', 'subject' containing sets of
    lowercase patterns that are already handled.
    """
    covered = {"from": set(), "to": set(), "subject": set()}
    rules = config.get("rules", [])
    if not rules:
        return covered

    def _extract_from_condition(cond: dict):
        for key, value in cond.items():
            key_upper = key.upper()
            if key_upper == "FROM":
                covered["from"].add(value.lower())
            elif key_upper == "TO":
                covered["to"].add(value.lower())
            elif key_upper == "SUBJECT":
                covered["subject"].add(value.lower())
            elif key_upper == "ANY":
                for sub in value:
                    if isinstance(sub, dict):
                        _extract_from_condition(sub)
            elif key_upper == "ALL":
                for sub in value:
                    if isinstance(sub, dict):
                        _extract_from_condition(sub)

    for rule in rules:
        if ignore_folders and rule.get("move_to") in ignore_folders:
            continue
        cond = rule.get("condition", {})
        if cond:
            _extract_from_condition(cond)

    return covered


def extract_existing_folders(config: dict) -> set[str]:
    """Get the set of folder names already used in mmuxer rules."""
    folders = set()
    for rule in config.get("rules", []):
        folder = rule.get("move_to")
        if folder:
            folders.add(folder)
    return folders


def is_message_covered(msg: MessageInfo, covered: dict) -> bool:
    """
    Check whether a message would be caught by existing rules.
    This is a heuristic — it checks if sender or recipient patterns appear
    in the covered sets. It doesn't fully replicate mmuxer's matching logic,
    but it's good enough for finding uncovered messages.
    """
    from_lower = msg.from_addr.lower()

    # Check exact FROM match
    if from_lower in covered["from"]:
        return True

    # Check partial FROM match (mmuxer may match on substrings)
    for pattern in covered["from"]:
        if pattern in from_lower or from_lower in pattern:
            return True

    # Check TO matches
    for to_addr in msg.to_addrs:
        to_lower = to_addr.lower()
        if to_lower in covered["to"]:
            return True
        for pattern in covered["to"]:
            if pattern in to_lower:
                return True

    # Check SUBJECT matches
    subj_lower = msg.subject.lower()
    for pattern in covered["subject"]:
        if pattern in subj_lower:
            return True

    return False


# ---------------------------------------------------------------------------
# IMAP connection and header fetching
# ---------------------------------------------------------------------------

def connect_imap(server: str, username: str, password: str, port: int = 993) -> imaplib.IMAP4_SSL:
    """Connect and authenticate to the IMAP server."""
    conn = imaplib.IMAP4_SSL(server, port)
    conn.login(username, password)
    return conn


def decode_header_value(raw: str) -> str:
    """Decode RFC 2047 encoded header values."""
    if raw is None:
        return ""
    parts = email.header.decode_header(raw)
    decoded = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return " ".join(decoded).strip()


def extract_email_address(from_header: str) -> tuple[str, str]:
    """
    Extract (email_address, display_name) from a From header.
    Returns lowercased email address.
    """
    decoded = decode_header_value(from_header)
    # Try to extract email from angle brackets
    match = re.search(r"<([^>]+)>", decoded)
    if match:
        addr = match.group(1).strip().lower()
        display = decoded[: match.start()].strip().strip('"').strip("'").strip()
        return addr, display
    # Bare email address
    addr = decoded.strip().lower()
    return addr, ""


def extract_to_addresses(to_header: str) -> list[str]:
    """Extract all email addresses from To/Cc headers."""
    if not to_header:
        return []
    decoded = decode_header_value(to_header)
    addrs = []
    for part in decoded.split(","):
        match = re.search(r"<([^>]+)>", part)
        if match:
            addrs.append(match.group(1).strip().lower())
        elif "@" in part:
            addrs.append(part.strip().lower())
    return addrs


def extract_list_id(headers: str) -> Optional[str]:
    """Extract List-Id value from raw headers."""
    match = re.search(r"^List-Id:\s*(.+)$", headers, re.MULTILINE | re.IGNORECASE)
    if match:
        value = match.group(1).strip()
        # Extract the <list-id> part if present
        id_match = re.search(r"<([^>]+)>", value)
        if id_match:
            return id_match.group(1).strip()
        return value
    return None


def fetch_inbox_headers(
    conn: imaplib.IMAP4_SSL,
    folder: str = "INBOX",
    limit: int = 500,
) -> list[MessageInfo]:
    """
    Fetch message headers from the specified folder.
    Returns a list of MessageInfo objects.
    """
    conn.select(folder, readonly=True)

    # Search for all messages
    status, data = conn.search(None, "ALL")
    if status != "OK":
        print(f"Failed to search {folder}: {status}", file=sys.stderr)
        return []

    uids = data[0].split()
    if not uids:
        print(f"No messages found in {folder}.")
        return []

    # Take the most recent N messages
    uids = uids[-limit:]
    print(f"Fetching headers for {len(uids)} messages from {folder}...")

    messages = []
    # Fetch in batches to avoid overwhelming the server
    batch_size = 100
    for i in range(0, len(uids), batch_size):
        batch = uids[i : i + batch_size]
        uid_range = b",".join(batch)

        # Fetch envelope headers + List-Id
        status, data = conn.fetch(
            uid_range,
            "(BODY.PEEK[HEADER.FIELDS (FROM TO CC SUBJECT DATE LIST-ID)])",
        )
        if status != "OK":
            print(f"Failed to fetch batch: {status}", file=sys.stderr)
            continue

        # data comes as pairs of (metadata, header_bytes) followed by closing paren
        j = 0
        batch_idx = 0
        while j < len(data):
            item = data[j]
            if isinstance(item, tuple) and len(item) == 2:
                uid = batch[batch_idx] if batch_idx < len(batch) else b"?"
                raw_headers = item[1]
                if isinstance(raw_headers, bytes):
                    raw_headers = raw_headers.decode("utf-8", errors="replace")

                # Parse with email module
                msg_obj = email.message_from_string(raw_headers)

                from_addr, from_display = extract_email_address(
                    msg_obj.get("From", "")
                )
                to_addrs = extract_to_addresses(msg_obj.get("To", ""))
                cc_addrs = extract_to_addresses(msg_obj.get("Cc", ""))
                subject = decode_header_value(msg_obj.get("Subject", ""))
                date = decode_header_value(msg_obj.get("Date", ""))
                list_id = extract_list_id(raw_headers)

                messages.append(
                    MessageInfo(
                        uid=uid.decode() if isinstance(uid, bytes) else str(uid),
                        from_addr=from_addr,
                        from_display=from_display,
                        to_addrs=to_addrs + cc_addrs,
                        subject=subject,
                        list_id=list_id,
                        date=date,
                    )
                )
                batch_idx += 1
            j += 1

        pct = min(100, int((i + len(batch)) / len(uids) * 100))
        print(f"  Progress: {pct}%", end="\r")

    print(f"\nFetched {len(messages)} messages.")
    return messages


# ---------------------------------------------------------------------------
# Analysis and grouping
# ---------------------------------------------------------------------------

def group_uncovered_messages(
    messages: list[MessageInfo], covered: dict
) -> list[SenderGroup]:
    """
    Filter out already-covered messages, then group remaining by sender.
    Returns sorted list of SenderGroup (highest count first).
    """
    uncovered = [m for m in messages if not is_message_covered(m, covered)]
    print(f"\n{len(uncovered)} of {len(messages)} messages are not covered by existing rules.")

    # Group by sender address
    by_sender: dict[str, list[MessageInfo]] = defaultdict(list)
    for m in uncovered:
        by_sender[m.from_addr].append(m)

    groups = []
    for addr, msgs in by_sender.items():
        subjects = list({m.subject for m in msgs})[:5]  # up to 5 unique subjects
        to_addrs = set()
        for m in msgs:
            to_addrs.update(m.to_addrs)
        list_ids = {m.list_id for m in msgs if m.list_id}
        groups.append(
            SenderGroup(
                from_addr=addr,
                from_display=msgs[0].from_display,
                count=len(msgs),
                sample_subjects=subjects,
                list_id=next(iter(list_ids), None),
                to_addrs=to_addrs,
            )
        )

    groups.sort(key=lambda g: g.count, reverse=True)
    return groups


# ---------------------------------------------------------------------------
# Rule suggestion logic
# ---------------------------------------------------------------------------

def suggest_folder_name(group: SenderGroup) -> str:
    """
    Heuristically suggest a folder name based on the sender info.
    Uses Category.Service convention.
    """
    addr = group.from_addr
    display = group.from_display
    domain = addr.split("@")[-1] if "@" in addr else addr

    # Use display name if available, otherwise domain
    if display:
        # Clean up common prefixes
        name = display
        for prefix in ["no-reply", "noreply", "notifications", "info", "support", "team"]:
            name = re.sub(rf"^{prefix}\s*[-@]\s*", "", name, flags=re.IGNORECASE)
        name = name.strip()
        if name:
            return name

    # Fall back to domain-based name
    # Strip common TLDs and extract meaningful part
    parts = domain.split(".")
    if len(parts) >= 2:
        name = parts[-2]  # e.g., "github" from "github.com"
    else:
        name = domain
    return name.capitalize()


def suggest_rule(group: SenderGroup) -> str:
    """Generate a YAML rule suggestion for a sender group."""
    folder = suggest_folder_name(group)

    # Determine the best matching strategy
    lines = []
    lines.append(f"  - move_to: {folder}")
    lines.append(f"    condition:")

    # If there's a subaddress in TO, prefer that
    subaddress_to = None
    for to in group.to_addrs:
        if "+" in to:
            subaddress_to = to
            break

    if subaddress_to:
        lines.append(f"      TO: {subaddress_to}")
    else:
        lines.append(f"      FROM: {group.from_addr}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interactive session
# ---------------------------------------------------------------------------

COLORS = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "red": "\033[31m",
    "reset": "\033[0m",
}


def color(text: str, *styles: str) -> str:
    prefix = "".join(COLORS.get(s, "") for s in styles)
    return f"{prefix}{text}{COLORS['reset']}"


def interactive_session(
    groups: list[SenderGroup],
    existing_folders: set[str],
    output_file: Optional[str] = None,
):
    """
    Walk through each sender group interactively.
    The user can accept, modify, skip, or quit.
    Accepted rules are collected and written to output or stdout at the end.
    """
    if not groups:
        print("\nNo uncovered sender groups found. Your rules are comprehensive!")
        return

    accepted_rules: list[str] = []
    total = len(groups)

    print(f"\n{'=' * 60}")
    print(color(f"  Found {total} uncovered sender groups", "bold"))
    print(f"{'=' * 60}")
    print()
    print("For each group, you can:")
    print(f"  {color('a', 'green')}ccept  — use the suggested rule as-is")
    print(f"  {color('f', 'yellow')}older  — accept but change the folder name")
    print(f"  {color('s', 'dim')}kip    — skip this group")
    print(f"  {color('q', 'red')}uit    — stop and output rules collected so far")
    print()

    for idx, group in enumerate(groups, 1):
        print(f"{color(f'[{idx}/{total}]', 'bold')} {color(group.from_addr, 'cyan')}")
        if group.from_display:
            print(f"  Display name: {group.from_display}")
        print(f"  Messages: {color(str(group.count), 'bold')}")
        if group.list_id:
            print(f"  List-Id: {group.list_id}")
        if group.to_addrs:
            relevant_to = [a for a in group.to_addrs if "+" in a]
            if relevant_to:
                print(f"  Subaddressed To: {', '.join(relevant_to)}")
        print(f"  Sample subjects:")
        for subj in group.sample_subjects[:3]:
            print(f"    • {subj[:80]}")

        suggestion = suggest_rule(group)
        print(f"\n  {color('Suggested rule:', 'green')}")
        for line in suggestion.split("\n"):
            print(f"    {line}")

        while True:
            choice = input(f"\n  [{color('a', 'green')}]ccept / [{color('f', 'yellow')}]older / [{color('s', 'dim')}]kip / [{color('q', 'red')}]uit: ").strip().lower()
            if choice in ("a", "accept", ""):
                accepted_rules.append(suggestion)
                print(f"  {color('✓ Accepted', 'green')}")
                break
            elif choice in ("f", "folder"):
                print(f"  Existing folders: {', '.join(sorted(existing_folders))}")
                new_folder = input("  Enter folder name: ").strip()
                if new_folder:
                    # Rebuild rule with new folder
                    modified = re.sub(
                        r"move_to: .+", f"move_to: {new_folder}", suggestion
                    )
                    accepted_rules.append(modified)
                    existing_folders.add(new_folder)
                    print(f"  {color('✓ Accepted with folder: ' + new_folder, 'green')}")
                else:
                    print("  No folder entered, using suggestion.")
                    accepted_rules.append(suggestion)
                break
            elif choice in ("s", "skip"):
                print(f"  {color('— Skipped', 'dim')}")
                break
            elif choice in ("q", "quit"):
                print(f"\n  Stopping early.")
                _output_rules(accepted_rules, output_file)
                return
            else:
                print("  Please enter a, f, s, or q.")

        print()

    _output_rules(accepted_rules, output_file)


def _output_rules(rules: list[str], output_file: Optional[str]):
    """Output the collected rules."""
    if not rules:
        print("\nNo rules were accepted.")
        return

    output = "\n".join(rules)

    if output_file:
        with open(output_file, "w") as f:
            f.write("# New rules generated by inbox_analyzer\n")
            f.write("# Append these to your mmuxer config under 'rules:'\n\n")
            f.write(output)
            f.write("\n")
        print(f"\n{color(f'Wrote {len(rules)} rules to {output_file}', 'green', 'bold')}")
    else:
        print(f"\n{'=' * 60}")
        print(color(f"  {len(rules)} new rules to add to your mmuxer config:", "bold"))
        print(f"{'=' * 60}")
        print()
        print(output)
        print()

    print(color("Copy the rules above into your mmuxer config.yaml under 'rules:'.", "dim"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze IMAP Inbox and suggest mmuxer rules for uncovered messages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config ~/mmuxer/config.yaml
  %(prog)s --config ~/mmuxer/config.yaml --server imap.example.com --username me@example.com
  %(prog)s --config ~/mmuxer/config.yaml --limit 1000 --output new_rules.yaml
  %(prog)s --config ~/mmuxer/config.yaml --min-count 3
        """,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to your mmuxer config.yaml",
    )
    parser.add_argument(
        "--server",
        help="IMAP server hostname (if not in mmuxer config)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=993,
        help="IMAP server port (default: 993)",
    )
    parser.add_argument(
        "--username",
        help="IMAP username/email (if not in mmuxer config)",
    )
    parser.add_argument(
        "--password",
        help="IMAP password (will prompt if not provided)",
    )
    parser.add_argument(
        "--env-file",
        metavar="FILE",
        help="path to a .env file with SERVER, USERNAME, PASSWORD, PORT",
    )
    parser.add_argument(
        "--folder",
        default="INBOX",
        help="IMAP folder to analyze (default: INBOX)",
    )
    parser.add_argument(
        "--ignore-folder",
        metavar="FOLDER",
        action="append",
        dest="ignore_folders",
        help="exclude rules targeting this folder from coverage check (repeatable)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum number of messages to fetch (default: 500, most recent)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum messages from a sender to suggest a rule (default: 2)",
    )
    parser.add_argument(
        "--output",
        help="Write accepted rules to this file instead of stdout",
    )

    args = parser.parse_args()

    # Load mmuxer config
    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_mmuxer_config(str(config_path))

    # Resolve connection settings
    settings = config.get("settings", {})

    # Load env file (default: .env next to the config file)
    env_file = Path(args.env_file).expanduser() if args.env_file else config_path.parent / ".env"
    env = load_env_file(env_file)

    server = args.server or env.get("server") or settings.get("server")
    username = args.username or env.get("username") or settings.get("username")
    if args.port == 993 and "port" in env:
        args.port = int(env["port"])

    if not server:
        server = input("IMAP server: ").strip()
    if not username:
        username = input("IMAP username: ").strip()

    password = args.password or env.get("password") or settings.get("password")
    if not password:
        password = getpass.getpass("IMAP password: ")

    # Extract existing rule patterns
    ignore_folders = set(args.ignore_folders or []) | {args.folder}
    covered = extract_covered_patterns(config, ignore_folders)
    existing_folders = extract_existing_folders(config)
    print(f"Loaded {sum(len(v) for v in covered.values())} patterns from existing rules.")
    print(f"Existing folders: {len(existing_folders)}")

    # Connect and fetch
    print(f"\nConnecting to {server}...")
    try:
        conn = connect_imap(server, username, password, args.port)
    except Exception as e:
        print(f"Failed to connect: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        messages = fetch_inbox_headers(conn, folder=args.folder, limit=args.limit)
    finally:
        try:
            conn.logout()
        except Exception:
            pass

    if not messages:
        print("No messages to analyze.")
        sys.exit(0)

    # Analyze
    groups = group_uncovered_messages(messages, covered)

    # Filter by minimum count
    groups = [g for g in groups if g.count >= args.min_count]
    if not groups:
        print(f"\nNo sender groups with {args.min_count}+ messages found.")
        one_offs = [g for g in group_uncovered_messages(messages, covered) if g.count == 1]
        if one_offs:
            print(f"({len(one_offs)} senders with only 1 message — use --min-count 1 to include them)")
        sys.exit(0)

    # Interactive rule suggestion
    interactive_session(groups, existing_folders, output_file=args.output)


if __name__ == "__main__":
    main()
