# Inbox Analyzer for mmuxer

## Context

Mykhailo uses [mmuxer](https://github.com/sapristi/mmuxer), a Python-based IMAP filtering client, to sort email on a remote IMAP server. mmuxer runs on an always-on intermediate machine (not the mail server itself — he has no server-side access) and connects as an IMAP client to move messages into folders based on YAML rules.

The problem: despite extensive rules, many emails remain unsorted in the Inbox because new subscriptions and senders appear over time. This script helps by analyzing the Inbox and proposing new mmuxer rules interactively.

**Privacy constraint**: No email data should ever leave the local machine. The script connects directly to IMAP and all processing happens locally. This is the entire reason the tool exists rather than using Claude directly against the mailbox.

## What exists

`inbox_analyzer.py` — a working Python script that:

1. Loads the mmuxer YAML config to extract existing FROM/TO/SUBJECT patterns and folder names
2. Connects to IMAP via SSL, fetches only headers (From, To, Cc, Subject, Date, List-Id) from Inbox
3. Filters out messages already covered by existing rules (heuristic substring matching)
4. Groups uncovered messages by sender address, sorted by message count
5. Interactively walks the user through each group: accept suggested rule, change folder name, skip, or quit
6. Outputs valid mmuxer YAML rule fragments ready to paste into the config

Dependencies: Python 3.10+, PyYAML (only non-stdlib dependency).

## mmuxer config format

The config is YAML. Rules live under a top-level `rules:` key. Each rule has `move_to:` (folder name using `Category.Service` hierarchy like `Admin.OVH`, `Apps.Music Production`) and `condition:` with matchers:

- `FROM: address` — match sender
- `TO: address` — match recipient (supports subaddressing like `apps+crossover@...`)
- `SUBJECT: text` — match subject
- `ANY:` — list of conditions, any must match (OR)
- `ALL:` — list of conditions, all must match (AND)
- `actions: [delete]` — alternative to `move_to`

Connection settings are under `settings:` (server, username, password). Settings can also come from environment variables.

Example rule:
```yaml
  - move_to: Apps.Music Production
    condition:
      ANY:
        - FROM: no-reply@news.ableton.com
        - TO: apps+ableton
        - FROM: news@arturia.com
```

## Known limitations and areas for improvement

- **Matching heuristic is loose**: `is_message_covered()` uses substring matching which may produce false negatives/positives. Could be improved to better replicate mmuxer's actual matching logic.
- **Folder name suggestion is naive**: `suggest_folder_name()` extracts from display name or domain. It doesn't attempt to fit into the user's existing `Category.Service` taxonomy. Could use existing folder hierarchy to suggest categories.
- **No List-Id based rules**: mmuxer may support List-Id matching but the current script only suggests FROM/TO rules. If mmuxer supports header matching, List-Id rules would be more robust for mailing lists.
- **Single sender grouping only**: doesn't detect patterns like "all emails from *.example.com" or "all emails to user+tag@...".
- **No persistent state**: running the script twice will re-suggest previously skipped groups. Could track dismissed senders.
- **No dry-run against existing mailbox**: can't verify whether proposed rules would have caught the right messages without actually running mmuxer.
- **Batch fetch could be optimized**: uses FETCH with sequence numbers rather than UID FETCH; could be more robust.
- **No OAuth support**: only app passwords. Fine for the current use case.
