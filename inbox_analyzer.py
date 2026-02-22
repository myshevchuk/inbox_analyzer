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
import email.utils
import getpass
import imaplib
import re
import sys
from collections import defaultdict
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
    from_addr: str  # normalized email address
    from_display: str  # display name
    to_addrs: list[str]
    subject: str
    list_id: Optional[str]
    date: str


@dataclass
class SenderGroup:
    """A cluster of messages sharing the same mailing list or sender address."""
    from_addr: str          # representative sender (most frequent, or only one)
    from_display: str       # display name of representative sender
    count: int
    sample_subjects: list[str]
    list_id: Optional[str] = None
    to_addrs: set[str] = field(default_factory=set)
    from_addrs: set[str] = field(default_factory=set)  # all senders in group


# ---------------------------------------------------------------------------
# mmuxer config parsing
# ---------------------------------------------------------------------------

def load_mmuxer_config(config_path: str) -> dict:
    """Load and return the mmuxer YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _walk_conditions(condition: dict):
    """Yield (key_upper, value) pairs from a mmuxer condition tree."""
    for key, value in condition.items():
        k = key.upper()
        if k in ("ANY", "ALL"):
            for sub in value:
                if isinstance(sub, dict):
                    yield from _walk_conditions(sub)
        else:
            yield k, value


def extract_covered_patterns(config: dict, ignore_folders: set[str] | None = None) -> dict:
    """
    Extract all FROM/TO/SUBJECT patterns from existing mmuxer rules.
    Returns a dict with keys 'from', 'to', 'subject' containing sets of
    lowercase patterns that are already handled.

    Rules whose move_to folder is in ignore_folders are skipped entirely,
    so their patterns do not count as coverage. This causes messages only
    matched by those rules to surface as uncovered.
    """
    covered = {"from": set(), "to": set(), "subject": set()}
    rules = config.get("rules", [])
    if not rules:
        return covered

    for rule in rules:
        if ignore_folders and rule.get("move_to") in ignore_folders:
            continue
        cond = rule.get("condition", {})
        if cond:
            for key, value in _walk_conditions(cond):
                if key.lower() in covered:
                    covered[key.lower()].add(value.lower())

    return covered


def build_rule_index(config: dict) -> list[tuple[str, set[str], set[str]]]:
    """
    Build a lookup index of existing rules for related-rule detection.
    Returns a list of (folder, from_patterns, to_patterns) — one entry per rule
    that has a move_to folder and at least one condition pattern.
    """
    index = []

    for rule in config.get("rules", []):
        folder = rule.get("move_to", "")
        if not folder:
            continue
        froms: set[str] = set()
        tos: set[str] = set()
        cond = rule.get("condition", {})
        if cond:
            for key, value in _walk_conditions(cond):
                if key == "FROM":
                    froms.add(value.lower())
                elif key == "TO":
                    tos.add(value.lower())
        if froms or tos:
            index.append((folder, froms, tos))

    return index


def _domains_related(d1: str, d2: str) -> bool:
    """Check if two domains are the same or one is a subdomain of the other."""
    if d1 == d2:
        return True
    return d1.endswith("." + d2) or d2.endswith("." + d1)


def find_related_rules(
    group: SenderGroup,
    rule_index: list[tuple[str, set[str], set[str]]],
) -> list[str]:
    """
    Return folder names of existing rules that appear related to this group.
    Matches by checking whether the sender domain or list_id domain overlaps
    (as a substring) with any existing FROM or TO pattern.
    """
    candidates: set[str] = set()
    if "@" in group.from_addr:
        candidates.add(group.from_addr.split("@")[-1].lower())
    if group.list_id:
        candidates.add(group.list_id.lower())

    related = []
    seen: set[str] = set()
    for folder, from_pats, to_pats in rule_index:
        if folder in seen:
            continue
        pat_domains = set()
        for pat in from_pats | to_pats:
            pat_domains.add(pat.split("@")[-1] if "@" in pat else pat)
        for domain in candidates:
            if any(_domains_related(domain, pd) for pd in pat_domains):
                related.append(folder)
                seen.add(folder)
                break

    return related


def get_ignored_to_patterns(config: dict, ignore_folders: set[str]) -> set[str]:
    """
    Return TO patterns that belong exclusively to ignored-folder rules.

    These patterns are used to suppress TO-based rule suggestions: if a
    message's TO address matches a pattern that was only in ignored rules,
    suggest_rule falls back to a FROM-based rule instead of recreating the
    rule that was already there.
    """
    all_covered = extract_covered_patterns(config)
    limited = extract_covered_patterns(config, ignore_folders)
    return all_covered["to"] - limited["to"]


def extract_existing_folders(config: dict) -> set[str]:
    """Get the set of folder names already used in mmuxer rules."""
    folders = set()
    for rule in config.get("rules", []):
        folder = rule.get("move_to")
        if folder:
            folders.add(folder)
    return folders



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
    return [addr.lower() for _, addr in email.utils.getaddresses([decoded]) if addr]


def extract_list_id(msg_obj) -> Optional[str]:
    """Extract List-Id value from a parsed email message object."""
    value = msg_obj.get("List-Id", "").strip()
    if not value:
        return None
    # Extract the <list-id> part if present
    id_match = re.search(r"<([^>]+)>", value)
    if id_match:
        return id_match.group(1).strip()
    return value


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

    seq_nums = data[0].split()
    if not seq_nums:
        print(f"No messages found in {folder}.")
        return []

    # Take the most recent N messages
    seq_nums = seq_nums[-limit:]
    print(f"Fetching headers for {len(seq_nums)} messages from {folder}...")

    messages = []
    # Fetch in batches to avoid overwhelming the server
    batch_size = 100
    for i in range(0, len(seq_nums), batch_size):
        batch = seq_nums[i : i + batch_size]
        seq_range = b",".join(batch)

        # Fetch envelope headers + List-Id
        status, data = conn.fetch(
            seq_range,
            "(BODY.PEEK[HEADER.FIELDS (FROM TO CC SUBJECT DATE LIST-ID)])",
        )
        if status != "OK":
            print(f"Failed to fetch batch: {status}", file=sys.stderr)
            continue

        # data comes as pairs of (metadata, header_bytes) followed by closing paren
        j = 0
        while j < len(data):
            item = data[j]
            if isinstance(item, tuple) and len(item) == 2:
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
                list_id = extract_list_id(msg_obj)

                messages.append(
                    MessageInfo(
                        from_addr=from_addr,
                        from_display=from_display,
                        to_addrs=to_addrs + cc_addrs,
                        subject=subject,
                        list_id=list_id,
                        date=date,
                    )
                )
            j += 1

        pct = min(100, int((i + len(batch)) / len(seq_nums) * 100))
        print(f"  Progress: {pct}%", end="\r")

    print(f"\nFetched {len(messages)} messages.")
    return messages


# ---------------------------------------------------------------------------
# Analysis and grouping
# ---------------------------------------------------------------------------

def _coverage_reason(msg: MessageInfo, covered: dict) -> Optional[str]:
    """Return a human-readable reason why a message is covered, or None if not covered."""
    from_lower = msg.from_addr.lower()

    if from_lower in covered["from"]:
        return f"FROM exact '{from_lower}'"
    for pattern in covered["from"]:
        if pattern in from_lower or from_lower in pattern:
            return f"FROM pattern '{pattern}'"

    for to_addr in msg.to_addrs:
        to_lower = to_addr.lower()
        if to_lower in covered["to"]:
            return f"TO exact '{to_lower}'"
        for pattern in covered["to"]:
            if pattern in to_lower:
                return f"TO pattern '{pattern}'"

    subj_lower = msg.subject.lower()
    for pattern in covered["subject"]:
        if pattern in subj_lower:
            return f"SUBJECT pattern '{pattern}'"

    return None


def group_uncovered_messages(
    messages: list[MessageInfo],
    covered: dict,
    ignore_to_patterns: set[str] | None = None,
    debug: bool = False,
) -> list[SenderGroup]:
    """
    Filter out already-covered messages, then group remaining by sender.
    Returns sorted list of SenderGroup (highest count first).

    ignore_to_patterns: TO address patterns from ignored-folder rules.
    Matching TO addresses are stripped from each SenderGroup so that
    suggest_rule uses the sender (FROM) rather than the TO subaddress,
    avoiding suggestions that duplicate already-existing ignored rules.
    """
    uncovered = []
    for m in messages:
        reason = _coverage_reason(m, covered)
        if reason:
            if debug:
                print(f"  [debug] covered:   {m.from_addr} | {reason}")
        else:
            if debug:
                print(f"  [debug] uncovered: {m.from_addr}")
            uncovered.append(m)
    print(f"\n{len(uncovered)} of {len(messages)} messages are not covered by existing rules.")

    # Group by list_id first (for mailing lists), then by sender address
    by_list_id: dict[str, list[MessageInfo]] = defaultdict(list)
    by_sender: dict[str, list[MessageInfo]] = defaultdict(list)
    for m in uncovered:
        if m.list_id:
            by_list_id[m.list_id].append(m)
            if debug:
                print(f"  [debug] list group  '{m.list_id}': {m.from_addr}")
        else:
            by_sender[m.from_addr].append(m)
            if debug:
                print(f"  [debug] sender group '{m.from_addr}'")

    groups = []

    def _build_to_addrs(msgs: list[MessageInfo]) -> set[str]:
        to_addrs = set()
        for m in msgs:
            to_addrs.update(m.to_addrs)
        if ignore_to_patterns:
            to_addrs = {
                t for t in to_addrs
                if not any(pat in t.lower() for pat in ignore_to_patterns)
            }
        return to_addrs

    for list_id, msgs in by_list_id.items():
        subjects = list({m.subject for m in msgs})[:5]
        from_addrs = {m.from_addr for m in msgs}
        # Pick the most frequent sender as the representative
        rep_addr = max(from_addrs, key=lambda a: sum(1 for m in msgs if m.from_addr == a))
        rep_display = next(m.from_display for m in msgs if m.from_addr == rep_addr)
        groups.append(
            SenderGroup(
                from_addr=rep_addr,
                from_display=rep_display,
                count=len(msgs),
                sample_subjects=subjects,
                list_id=list_id,
                to_addrs=_build_to_addrs(msgs),
                from_addrs=from_addrs,
            )
        )

    for addr, msgs in by_sender.items():
        subjects = list({m.subject for m in msgs})[:5]
        groups.append(
            SenderGroup(
                from_addr=addr,
                from_display=msgs[0].from_display,
                count=len(msgs),
                sample_subjects=subjects,
                list_id=None,
                to_addrs=_build_to_addrs(msgs),
                from_addrs={addr},
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
    subaddress_to = next((to for to in sorted(group.to_addrs) if "+" in to), None)

    if subaddress_to:
        lines.append(f"      TO: {subaddress_to}")
    elif len(group.from_addrs) > 1:
        lines.append(f"      ANY:")
        for addr in sorted(group.from_addrs):
            lines.append(f"        - FROM: {addr}")
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
_USE_COLOR = sys.stdout.isatty()


def color(text: str, *styles: str) -> str:
    if not _USE_COLOR:
        return text
    prefix = "".join(COLORS.get(s, "") for s in styles)
    return f"{prefix}{text}{COLORS['reset']}"


def interactive_session(
    groups: list[SenderGroup],
    existing_folders: set[str],
    rule_index: list[tuple[str, set[str], set[str]]],
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
        related = find_related_rules(group, rule_index)
        if related:
            for i, folder in enumerate(related, 1):
                print(color(f"  \u21b3 [{i}] Related existing rule: {folder}", "yellow"))

        related_hint = ""
        if related:
            n = len(related)
            nums = "1" if n == 1 else f"1-{n}"
            related_hint = f" / [{color(nums, 'yellow')}] use related"
        while True:
            choice = input(f"\n  [{color('a', 'green')}]ccept / [{color('f', 'yellow')}]older{related_hint} / [{color('s', 'dim')}]kip / [{color('q', 'red')}]uit: ").strip().lower()
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
            elif choice.isdigit() and 1 <= int(choice) <= len(related):
                chosen_folder = related[int(choice) - 1]
                rule = re.sub(r"move_to: .+", f"move_to: {chosen_folder}", suggestion)
                accepted_rules.append(rule)
                existing_folders.add(chosen_folder)
                print(color(f"  ✓ Accepted with folder: {chosen_folder}", "green"))
                break
            else:
                opts = "a, f, s, q" + (f", or 1-{len(related)}" if related else "")
                print(f"  Please enter {opts}.")

        print()

    _output_rules(accepted_rules, output_file)


def _output_rules(rules: list[str], output_file: Optional[str]):
    """Output the collected rules."""
    if not rules:
        print("\nNo rules were accepted.")
        return

    output = "\n\n".join(rules)

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
        "--debug",
        action="store_true",
        default=False,
        help="Print per-message coverage decisions and group assignments.",
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
    ignore_to_patterns = get_ignored_to_patterns(config, ignore_folders)
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
    all_groups = group_uncovered_messages(messages, covered, ignore_to_patterns, debug=args.debug)

    # Filter by minimum count
    groups = [g for g in all_groups if g.count >= args.min_count]
    if not groups:
        print(f"\nNo sender groups with {args.min_count}+ messages found.")
        one_offs = [g for g in all_groups if g.count == 1]
        if one_offs:
            print(f"({len(one_offs)} senders with only 1 message — use --min-count 1 to include them)")
        sys.exit(0)

    # Interactive rule suggestion
    rule_index = build_rule_index(config)
    interactive_session(groups, existing_folders, rule_index, output_file=args.output)


if __name__ == "__main__":
    main()
