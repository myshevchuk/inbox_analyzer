# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`inbox_analyzer.py` is a single-file Python script that connects to IMAP, fetches message headers, and interactively proposes [mmuxer](https://github.com/sapristi/mmuxer) filter rules for senders not yet covered by an existing config.

**Privacy invariant:** No email data should ever leave the local machine. The script connects directly to IMAP and processes everything locally — this is the reason the tool exists rather than delegating to a cloud service.

## Commands

```sh
# Run
python "inbox_analyzer.py" --config ~/mmuxer/config.yaml

# Tests
pytest "test_grouping.py"

# Run a single test
pytest "test_grouping.py" -k "test_name"

# Lint
ruff check "inbox_analyzer.py"
```

**Dependencies:** `pip install pyyaml pytest` (only non-stdlib dependencies; pytest is for development only).

## Development

Development is test-driven. Write or update tests in `test_grouping.py` before implementing changes.

Check `TODO.md` at the start of each session for known bugs. If bugs or features come up during a conversation but are not immediately addressed, add them to `TODO.md` before closing the session.

**Documentation:** `README.md` is user-facing documentation — update it when CLI behavior or output changes. Use inline comments in `inbox_analyzer.py` for implementation decisions and non-obvious logic.

## Architecture

Everything lives in two files: `inbox_analyzer.py` (application) and `test_grouping.py` (tests). No package structure, no build step.

### Pipeline

```
load_mmuxer_config()
  → extract_covered_patterns()       # build coverage set from existing rules
  → fetch_inbox_headers()            # IMAP SSL, headers only, batched 100/request
  → _coverage_reason() per message   # filter already-handled messages
  → classify_and_group_emails()
      → classify_message()           # per-email: token extraction, anchor detection
      → group_classified_messages()  # merge into MessageGroup, cross-link TO/FROM
  → interactive_session()            # TUI: accept / rename / skip / merge / quit
  → _output_rules()                  # write YAML to stdout or --output file
```

### Key Data Structures

- **`MessageInfo`** — raw header data: `from_addr`, `from_display`, `to_addrs`, `subject`, `list_id`, `date`
- **`MessageFeatures`** — classification result: adds `group_key`, `anchor_type`, `anchor_tokens`, `suggested_destination`, `to_alias`
- **`MessageGroup`** — accumulated group: representative sender, message count, sample subjects, all addresses, cross-links to related groups

### Grouping Priority

1. **TO anchor** — delivered to a subaddress (`user+tag@domain`); key = `TO:local+tag`
2. **LIST anchor** — has `List-Id` header; key = `LIST:<normalized-list-id>`
3. **FROM anchor** — plain sender; key = `FROM:addr@domain`

### Folder Name Suggestion

Domain, display name, and subject are tokenized and stopwords removed. `find_strong_signals()` finds tokens present in ≥2 of the 3 sources. `match_tokens_to_rule()` scores against existing folder leaf names. The `to_alias` field (e.g. `"apps"` from `apps+spotify@domain`) is used as a category prefix (e.g. `Apps.Spotify`). Folder suggestion uses heuristic token matching only and will not use a machine learning approach.

### mmuxer Rule Format

Rules go under the top-level `rules:` key. Supported conditions: `FROM:`, `TO:`, `SUBJECT:`, `ANY:` (OR), `ALL:` (AND). mmuxer does **not** support `List-Id` as a condition — mailing lists with rotating senders are emitted as `ANY: [FROM: ...]`.

## Limitations

- **No persistent state:** running the script twice will re-suggest previously skipped groups.
- **No OAuth:** only app passwords are supported.
