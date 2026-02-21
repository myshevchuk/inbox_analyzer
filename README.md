# inbox_analyzer

Analyzes your IMAP Inbox and interactively proposes [mmuxer](https://github.com/sapristi/mmuxer) filter rules for senders not yet covered by your config. All processing happens locally — no email data leaves your machine.

## How it works

1. Loads your mmuxer `config.yaml` to extract existing rules and folder names
2. Connects to IMAP and fetches message headers (From, To, Subject, Date) from Inbox
3. Filters out messages already matched by existing rules
4. Groups remaining messages by sender (or by List-Id for mailing lists), sorted by frequency
5. Walks you through each group interactively — accept, rename the folder, skip, or quit
6. Outputs valid mmuxer YAML rule fragments ready to paste into your config

## Install

**Dependencies:** Python 3.10+, [PyYAML](https://pypi.org/project/PyYAML/)

```sh
curl -o ~/bin/inbox_analyzer.py https://raw.githubusercontent.com/myshevchuk/inbox_analyzer/refs/heads/main/inbox_analyzer.py
chmod +x ~/bin/inbox_analyzer.py
pip install pyyaml
```

## Usage

```sh
inbox_analyzer.py --config ~/mmuxer/config.yaml
```

Credentials are resolved in order: CLI flags → `.env` file → mmuxer config → interactive prompt.

### Env file

By default the script looks for `.env` next to your `config.yaml`. Keys match mmuxer's own setting names (case-insensitive):

```sh
server=imap.example.com
username=me@example.com
password=secret
port=993
```

Use `--env-file` to point to a different path.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | *(required)* | Path to mmuxer `config.yaml` |
| `--env-file` | `.env` next to config | Path to env file with credentials |
| `--server` | from env/config | IMAP server hostname |
| `--port` | `993` | IMAP port |
| `--user` | from env/config | IMAP username |
| `--password` | *(prompted)* | IMAP password |
| `--folder` | `INBOX` | Folder to analyze |
| `--ignore-folder` | *(none)* | Exclude rules targeting this folder from coverage check (repeatable) |
| `--limit` | `500` | Max messages to fetch |
| `--min-count` | `2` | Min messages from a sender to suggest a rule |
| `--output` | stdout | Write accepted rules to a file |

### Rule suggestion logic

When a message was sent to a subaddress (`user+tag@domain`), the script suggests a `TO:` rule rather than `FROM:`, since the subaddress is typically a more stable identifier than the sender. Otherwise it falls back to `FROM:`.

For mailing lists grouped by List-Id (which may rotate sender addresses), the script generates an `ANY: [FROM: ...]` rule covering all observed sender addresses in that list.

**Exception:** if the subaddress pattern belongs to a rule targeting an ignored folder, the `TO:` suggestion is suppressed and `FROM:` is used instead — to avoid recreating a rule that was explicitly ignored.

### Ignoring catch-all rules

If broad catch-all rules hide senders that deserve more specific rules, exclude them from the coverage check with `--ignore-folder`:

```sh
inbox_analyzer.py --config ~/mmuxer/config.yaml --ignore-folder Misc
```

The folder passed via `--folder` is always ignored automatically.

### Examples

```sh
# Analyze last 1000 messages, save accepted rules to a file
inbox_analyzer.py --config ~/mmuxer/config.yaml --limit 1000 --output new_rules.yaml

# Analyze a non-inbox folder, also ignoring a catch-all
inbox_analyzer.py --config ~/mmuxer/config.yaml --folder Newsletter --ignore-folder Misc
```
