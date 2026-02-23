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
from dataclasses import dataclass, field, fields as dataclass_fields
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
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            env[key.strip().lower()] = value
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
    group_key: str = ""                    # unique key: "TO:local+tag" or "FROM:addr" or "LIST:list_id"
    anchor_type: str = ""                  # "TO", "FROM", "LIST"
    anchor_tokens: list[str] = field(default_factory=list)  # tokens for rule matching
    suggested_destination: Optional[str] = None             # matched folder from sender_index
    related_group_keys: list[str] = field(default_factory=list)  # keys of related groups
    recipient_category: str = ""  # local part of TO alias on mydomain (if not primary inbox)


@dataclass
class MessageFeatures:
    """Per-email classification result from classify_message."""
    # raw (echoed from MessageInfo for SenderGroup construction)
    from_addr: str
    from_display: str
    list_id: Optional[str]    # raw value (not normalized) — needed for SenderGroup display
    subject: str
    to_addrs: list[str]       # list[str] matching MessageInfo.to_addrs; set conversion happens in group_classified_messages
    # classification
    group_key: str            # "TO:apps+spotify" | "LIST:list.example.com" | "FROM:foo@bar.com"
    anchor_type: str          # "TO" | "LIST" | "FROM"
    anchor_tokens: list[str]
    suggested_destination: Optional[str]
    recipient_hint: Optional[str]   # debug/carry-forward only; not used in grouping
    recipient_category: str
    # intermediate tokens (kept for debugging/future use)
    domain_tokens: list[str]
    display_tokens: list[str]
    subject_tokens: list[str]


# ---------------------------------------------------------------------------
# mmuxer config parsing
# ---------------------------------------------------------------------------

def load_mmuxer_config(config_path: str) -> dict:
    """Load and return the mmuxer YAML config."""
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Failed to parse config: {e}", file=sys.stderr)
        sys.exit(1)


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


def extract_mydomain(config: dict) -> Optional[str]:
    """Extract the domain part from the username in config settings.

    Returns the domain (part after '@') from config['settings']['username'],
    or None if the settings key is missing, the username key is missing, or
    the username contains no '@'.
    """
    settings = config.get("settings")
    if not settings:
        return None
    username = settings.get("username")
    if not username or "@" not in username:
        return None
    return username.split("@", 1)[1]


def build_sender_index(config: dict) -> dict[str, str]:
    """Build a mapping from lowercase folder leaf name to full folder path.

    Iterates over rules with a move_to key and extracts the leaf segment
    (the last dot-separated component of the folder path).  The leaf is
    lowercased and used as the key.  On collision (two folders share the
    same lowercase leaf), the alphabetically first full path is kept.

    Example:
        rules with move_to values ["Apps.Spotify", "Apps.Music Production"]
        ->  {"spotify": "Apps.Spotify", "music production": "Apps.Music Production"}
    """
    index: dict[str, str] = {}
    for rule in config.get("rules", []):
        folder = rule.get("move_to")
        if not folder:
            continue
        leaf = folder.split(".")[-1].lower()
        if leaf not in index or folder < index[leaf]:
            index[leaf] = folder
    return index


# ---------------------------------------------------------------------------
# IMAP connection and header fetching
# ---------------------------------------------------------------------------

def connect_imap(server: str, username: str, password: str, port: int = 993) -> imaplib.IMAP4_SSL:
    """Connect and authenticate to the IMAP server."""
    conn = imaplib.IMAP4_SSL(server, port, timeout=30)
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


def _normalize_list_id(list_id: str) -> str:
    """Strip leading purely-numeric dot-separated labels from a List-Id.

    Some mailing lists embed per-message numbers as prefixes, e.g.
    '16138.list-id.www.example.de'. Stripping the numeric prefix allows
    messages from the same list to be grouped together.
    """
    parts = list_id.split(".")
    while parts and parts[0].isdigit():
        parts.pop(0)
    return ".".join(parts) if parts else list_id


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
        print(f"  Progress: {pct:3d}%    ", end="\r")

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


STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "can",
    "could", "your", "you", "my", "our", "we", "us", "me", "he", "she",
    "it", "they", "them", "this", "that", "these", "new", "re", "fw",
    "fwd", "no-reply", "noreply", "newsletter", "update", "notification",
    "alert", "info", "hello", "hi", "dear", "please", "thanks", "welcome",
    "confirm", "verify", "action", "required", "notifications",
})


_KNOWN_TLDS = {
    # generic
    "com", "net", "org", "edu", "gov", "mil", "int", "biz", "info", "pro",
    "name", "mobi", "coop", "aero", "jobs", "travel", "museum",
    # new generic
    "io", "co", "ai", "app", "dev", "cloud", "online", "site", "web",
    "store", "shop", "blog", "news", "media", "tech", "digital", "social",
    "group", "team", "agency", "studio", "design", "space", "zone", "works",
    "solutions", "services", "systems", "network", "company", "global",
    "world", "live", "plus", "pro", "xyz", "top", "club", "tools", "fun",
    # email infra
    "email", "mail",
    # EU and European national
    "eu", "uk", "de", "fr", "it", "es", "nl", "pl", "se", "no", "dk",
    "fi", "pt", "be", "at", "ch", "cz", "sk", "hu", "ro", "bg", "hr",
    "si", "lt", "lv", "ee", "ie", "lu", "is", "gr", "ua",
    # other common national
    "us", "ca", "au", "nz", "jp", "cn", "ru", "br", "in", "mx",
}


def tokenize(text: str) -> list[str]:
    """Split text into meaningful lowercase tokens, filtering stopwords."""
    tokens = re.split(r'[^a-z0-9]+', text.lower())
    return [t for t in tokens if len(t) >= 2 and t not in STOPWORDS]


def extract_domain_tokens(domain: str) -> list[str]:
    """Extract meaningful tokens from a domain name, skipping known TLDs."""
    parts = [p for p in domain.lower().split(".") if p and p not in _KNOWN_TLDS]
    return tokenize(parts[-1]) if parts else []


def find_strong_signals(
    domain_tokens: list[str],
    display_tokens: list[str],
    subject_tokens: list[str],
) -> list[str]:
    """Return tokens that appear in at least 2 of the 3 input lists."""
    domain_set = set(domain_tokens)
    display_set = set(display_tokens)
    subject_set = set(subject_tokens)
    strong = (domain_set & display_set) | (domain_set & subject_set) | (display_set & subject_set)
    seen: set[str] = set()
    result = []
    for t in domain_tokens + display_tokens:
        if t in strong and t not in seen:
            seen.add(t)
            result.append(t)
    return result


def match_tokens_to_rule(
    tokens: list[str],
    sender_index: dict[str, str],
    recipient_hint: Optional[str] = None,
) -> Optional[str]:
    """Return the folder path from sender_index that best matches tokens.

    For each key in sender_index, split it into sub-tokens by non-alphanumeric
    characters and count how many input tokens appear in those sub-tokens.
    The key with the highest count (score > 0) wins.

    Tie-breaking (equal scores):
    1. Prefer the folder whose category (everything before the last '.') matches
       recipient_hint (case-insensitive).
    2. Alphabetical on folder path as final tiebreaker.
    """
    if not tokens:
        return None

    token_set = set(tokens)
    best_folder: Optional[str] = None
    best_score = 0

    for key, folder in sender_index.items():
        sub_tokens = set(re.split(r"[^a-z0-9]+", key.lower()))
        sub_tokens.discard("")
        score = len(token_set & sub_tokens)
        if score == 0:
            continue

        if score > best_score:
            best_score = score
            best_folder = folder
        elif score == best_score:
            # Tie-breaking: recipient_hint category match, then alphabetical
            assert best_folder is not None
            current_category = best_folder.rsplit(".", 1)[0].lower() if "." in best_folder else best_folder.lower()
            candidate_category = folder.rsplit(".", 1)[0].lower() if "." in folder else folder.lower()
            hint = recipient_hint.lower() if recipient_hint else None

            current_hint_match = hint is not None and current_category == hint
            candidate_hint_match = hint is not None and candidate_category == hint

            if candidate_hint_match and not current_hint_match:
                best_folder = folder
            elif not candidate_hint_match and current_hint_match:
                pass  # keep current
            else:
                # Both match hint or neither does — alphabetical
                if folder < best_folder:
                    best_folder = folder

    return best_folder




def classify_message(
    msg: MessageInfo,
    mydomain: Optional[str],
    primary_local: Optional[str],
    sender_index: dict[str, str],
) -> MessageFeatures:
    """Classify a single MessageInfo into a MessageFeatures instance.

    Extracts tokens from domain, display name, and subject; detects subaddress
    anchoring; matches against sender_index for a suggested destination; and
    assigns a fully-resolved group_key and anchor_type.
    """

    anchor_type = ""
    group_key = ""
    recipient_category = ""
    recipient_hint = None
    tag_tokens: list[str] = []

    # Step 1: scan TO addresses for subaddress and recipient category in one pass.
    # NOTE: only TO addresses are checked; CC is not yet in MessageInfo.
    # Multiple subaddressed TO addresses is a theoretical edge case and
    # is handled by taking the first match.
    if mydomain:
        mydomain_lower = mydomain.lower()
        primary_lower = (primary_local or "").lower()
        for addr in msg.to_addrs:
            addr_lower = addr.lower()
            if not addr_lower.endswith("@" + mydomain_lower):
                continue
            parts = tuple(addr_lower.rsplit("@", 1)[0].split("+"))
            base_local = parts[0]
            if not anchor_type and len(parts) > 1:
                anchor_type = "TO"
                recipient_hint = base_local
                group_key = "TO:" + "+".join(parts)
                tag_tokens = list(parts[1:])
            if not recipient_category and base_local and base_local != primary_lower:
                recipient_category = base_local
            if anchor_type and recipient_category:
                break

    # Step 3: token extraction
    sender_domain = msg.from_addr.split("@")[-1] if "@" in msg.from_addr else msg.from_addr
    domain_tokens = extract_domain_tokens(sender_domain)
    display_tokens = tokenize(msg.from_display)
    subject_tokens = tokenize(msg.subject) if msg.subject else []
    strong = find_strong_signals(domain_tokens, display_tokens, subject_tokens)
    if tag_tokens:
        anchor_tokens = tag_tokens + [t for t in (strong or domain_tokens) if t not in tag_tokens]
    else:
        anchor_tokens = strong if strong else domain_tokens

    # Step 4: rule matching
    suggested_destination = match_tokens_to_rule(anchor_tokens, sender_index, recipient_hint)

    # Step 5: group key finalization
    if anchor_type == "TO":
        pass  # group_key already set; subaddress takes priority over list_id
    elif msg.list_id:
        anchor_type = "LIST"
        group_key = f"LIST:{_normalize_list_id(msg.list_id)}"
    else:
        anchor_type = "FROM"
        group_key = f"FROM:{msg.from_addr}"

    assert group_key, f"group_key must be non-empty for {msg.from_addr}"

    return MessageFeatures(
        from_addr=msg.from_addr,
        from_display=msg.from_display,
        list_id=msg.list_id,
        subject=msg.subject,
        to_addrs=msg.to_addrs,
        group_key=group_key,
        anchor_type=anchor_type,
        anchor_tokens=anchor_tokens,
        suggested_destination=suggested_destination,
        recipient_hint=recipient_hint,
        recipient_category=recipient_category,
        domain_tokens=domain_tokens,
        display_tokens=display_tokens,
        subject_tokens=subject_tokens,
    )


def group_classified_messages(
    features: list[MessageFeatures],
) -> list[SenderGroup]:
    """Group a list of MessageFeatures into SenderGroup objects with cross-links.

    Accumulates features by group_key (first-message-wins for anchor fields),
    runs a cross-group linking pass between TO and FROM groups sharing anchor
    tokens, and returns groups sorted descending by count.
    """
    groups: dict[str, SenderGroup] = {}

    for feature in features:
        group_key = feature.group_key
        if group_key in groups:
            existing = groups[group_key]
            existing.count += 1
            existing.from_addrs.add(feature.from_addr)
            existing.to_addrs.update(feature.to_addrs)
            if len(existing.sample_subjects) < 5 and feature.subject not in existing.sample_subjects:
                existing.sample_subjects.append(feature.subject)
        else:
            groups[group_key] = SenderGroup(
                from_addr=feature.from_addr,
                from_display=feature.from_display,
                count=1,
                sample_subjects=[feature.subject] if feature.subject else [],
                list_id=feature.list_id,
                to_addrs=set(feature.to_addrs),
                from_addrs={feature.from_addr},
                group_key=group_key,
                anchor_type=feature.anchor_type,
                anchor_tokens=feature.anchor_tokens,
                suggested_destination=feature.suggested_destination,
                related_group_keys=[],
                recipient_category=feature.recipient_category,
            )

    # Cross-group linking pass: TO groups <-> FROM groups sharing anchor tokens
    to_groups = [g for g in groups.values() if g.anchor_type == "TO"]
    from_groups = [g for g in groups.values() if g.anchor_type == "FROM"]

    for to_group in to_groups:
        to_token_set = set(to_group.anchor_tokens)
        for from_group in from_groups:
            from_token_set = set(from_group.anchor_tokens)
            if to_token_set & from_token_set:
                if from_group.group_key not in to_group.related_group_keys:
                    to_group.related_group_keys.append(from_group.group_key)
                if to_group.group_key not in from_group.related_group_keys:
                    from_group.related_group_keys.append(to_group.group_key)

    result = list(groups.values())
    result.sort(key=lambda g: g.count, reverse=True)
    return result


def classify_and_group_emails(
    uncovered_emails: list[MessageInfo],
    config: dict,
    mydomain: Optional[str],
    primary_local: Optional[str] = None,
) -> list[SenderGroup]:
    """
    Classify and group uncovered emails into SenderGroup objects with enriched
    metadata: group key, anchor type, anchor tokens, suggested destination, and
    cross-group relationships.

    Per-email pipeline:
    1. Subaddress check — detect TO subaddressing on mydomain
    2. Token extraction — domain, display, subject tokens combined via find_strong_signals
    3. Rule matching — look up anchor_tokens in sender_index for a suggested destination
    4. Group key assignment — based on subaddress, list_id, or sender address
    5. Group accumulation — merge into existing group or create a new one
    After loop: cross-group linking pass between TO and FROM groups sharing tokens.
    Returns groups sorted descending by count.
    """
    sender_index = build_sender_index(config)
    features = [
        classify_message(msg, mydomain, primary_local, sender_index)
        for msg in uncovered_emails
    ]
    return group_classified_messages(features)


# ---------------------------------------------------------------------------
# Rule suggestion logic
# ---------------------------------------------------------------------------

def suggest_folder_name(group: SenderGroup) -> str:
    """Heuristically suggest a folder name based on the sender info."""
    if group.suggested_destination:
        return group.suggested_destination

    # For TO groups with subaddress: category=prefix, service=tag
    if group.anchor_type == "TO" and "+" in group.group_key:
        local_part = group.group_key[len("TO:"):]
        prefix, tag = local_part.split("+", 1)
        service = tag.capitalize()
        return f"{prefix.capitalize()}.{service}"

    # Category prefix from non-primary TO alias (e.g. forma@domain.com → "Forma")
    category_prefix = group.recipient_category.capitalize() if group.recipient_category else ""

    addr = group.from_addr
    display = group.from_display
    if not addr and not display:
        return f"{category_prefix}.Uncategorized" if category_prefix else "Uncategorized"
    domain = addr.split("@")[-1] if "@" in addr else addr

    if display:
        name = display
        for prefix_str in ["no-reply", "noreply", "notifications", "info", "support", "team"]:
            name = re.sub(rf"^{prefix_str}\s*[-@]\s*", "", name, flags=re.IGNORECASE)
        name = name.strip()
        if name:
            return f"{category_prefix}.{name}" if category_prefix else name

    parts = domain.split(".")
    name = parts[-2] if len(parts) >= 2 else domain
    name = name.capitalize()
    return f"{category_prefix}.{name}" if category_prefix else name


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
    debug: bool = False,
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
    merged_keys: set[str] = set()

    # Build a lookup map from group_key to SenderGroup for merge support
    groups_by_key: dict[str, SenderGroup] = {g.group_key: g for g in groups if g.group_key}

    print(f"\n{'=' * 60}")
    print(color(f"  Found {total} uncovered sender groups", "bold"))
    print(f"{'=' * 60}")
    print()

    if debug:
        print(color("  [debug] Group summary:", "dim"))
        print(color(f"    {'#':>3}  {'anchor':<40} {'n':>4}  destination / related", "dim"))
        print(color(f"    {'-'*3}  {'-'*40} {'-'*4}  {'-'*30}", "dim"))
        for i, g in enumerate(groups, 1):
            anchor = g.group_key or "?"
            dest = f"-> {g.suggested_destination}" if g.suggested_destination else ""
            related = f"(related: {', '.join(g.related_group_keys)})" if g.related_group_keys else ""
            suffix = dest or related
            print(color(f"    {i:>3}  {anchor:<40} {g.count:>4}  {suffix}", "dim"))
        print()

    print("For each group, you can:")
    print(f"  {color('a', 'green')}ccept  — use the suggested rule as-is")
    print(f"  {color('f', 'yellow')}older  — accept but change the folder name")
    print(f"  {color('s', 'dim')}kip    — skip this group")
    print(f"  {color('q', 'red')}uit    — stop and output rules collected so far")
    print()

    try:
        for idx, group in enumerate(groups, 1):
            if group.group_key in merged_keys:
                continue

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

            # Show related group keys if any
            if group.related_group_keys:
                print(f"  Related groups: {', '.join(group.related_group_keys)}")

            if debug:
                print(color("  [debug]", "dim"))
                for f in dataclass_fields(group):
                    val = getattr(group, f.name)
                    if isinstance(val, (set, list)):
                        val = sorted(val) if isinstance(val, set) else val
                    print(color(f"    {f.name + ':':<26} {val!r}", "dim"))

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

            has_related_groups = bool(group.related_group_keys)
            merge_hint = f" / [{color('m', 'cyan')}]erge" if has_related_groups else ""

            while True:
                choice = input(f"\n  [{color('a', 'green')}/↵]ccept / [{color('f', 'yellow')}]older{related_hint}{merge_hint} / [{color('s', 'dim')}]kip / [{color('q', 'red')}]uit: ").strip().lower()
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
                elif choice in ("m", "merge") and has_related_groups:
                    folder = suggest_folder_name(group)
                    # Extract condition lines from suggestion (everything after "condition:")
                    current_condition_lines = []
                    in_condition = False
                    for line in suggestion.split("\n"):
                        stripped = line.strip()
                        if stripped == "condition:":
                            in_condition = True
                            continue
                        if in_condition:
                            current_condition_lines.append(f"          {stripped}")

                    # Build ANY conditions list
                    any_items: list[str] = []
                    any_items.extend(current_condition_lines)

                    for rkey in group.related_group_keys:
                        related_group = groups_by_key.get(rkey)
                        if related_group is not None:
                            rel_suggestion = suggest_rule(related_group)
                            in_cond = False
                            for line in rel_suggestion.split("\n"):
                                stripped = line.strip()
                                if stripped == "condition:":
                                    in_cond = True
                                    continue
                                if in_cond:
                                    any_items.append(f"          {stripped}")

                    merged_lines = [
                        f"  - move_to: {folder}",
                        f"    condition:",
                        f"      ANY:",
                    ]
                    for item in any_items:
                        merged_lines.append(f"        - {item.strip()}")

                    merged_rule = "\n".join(merged_lines)
                    accepted_rules.append(merged_rule)

                    # Mark related groups as merged
                    for rkey in group.related_group_keys:
                        merged_keys.add(rkey)

                    print(f"  {color('✓ Merged and accepted', 'green')}")
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
                    if has_related_groups:
                        opts += ", or m"
                    print(f"  Please enter {opts}.")

            print()

    except KeyboardInterrupt:
        print(f"\n\n  Interrupted.")

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
    # Filter uncovered messages
    uncovered = []
    for m in messages:
        reason = _coverage_reason(m, covered)
        if reason:
            if args.debug:
                print(f"  [debug] covered:   {m.from_addr} | {reason}")
        else:
            if args.debug:
                print(f"  [debug] uncovered: {m.from_addr}")
            uncovered.append(m)
    print(f"\n{len(uncovered)} of {len(messages)} messages are not covered by existing rules.")

    mydomain = (username.split("@", 1)[1] if username and "@" in username else None) or extract_mydomain(config)
    primary_local = username.split("@")[0].lower() if username and "@" in username else None
    all_groups = classify_and_group_emails(uncovered, config, mydomain, primary_local=primary_local)

    # Strip ignored TO patterns from group to_addrs
    if ignore_to_patterns:
        for g in all_groups:
            g.to_addrs = {
                t for t in g.to_addrs
                if not any(pat in t.lower() for pat in ignore_to_patterns)
            }

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
    interactive_session(groups, existing_folders, rule_index, output_file=args.output, debug=args.debug)


if __name__ == "__main__":
    main()
