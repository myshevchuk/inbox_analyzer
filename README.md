# inbox_analyzer

Analyzes your IMAP Inbox and interactively proposes [mmuxer](https://github.com/sapristi/mmuxer) filter rules for senders not yet covered by your config. All processing happens locally — no email data leaves your machine.

## How it works

1. Loads your mmuxer `config.yaml` to extract existing rules and folder names
2. Connects to IMAP and fetches message headers (From, To, Subject, Date) from Inbox
3. Filters out messages already matched by existing rules
4. Groups remaining messages by sender, sorted by frequency
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

Connection settings (`server`, `username`) are read from your mmuxer config. Password is prompted interactively if not present in config.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | *(required)* | Path to mmuxer `config.yaml` |
| `--server` | from config | IMAP server hostname |
| `--port` | `993` | IMAP port |
| `--user` | from config | IMAP username |
| `--password` | *(prompted)* | IMAP password |
| `--folder` | `INBOX` | Folder to analyze |
| `--limit` | `500` | Max messages to fetch |
| `--min-count` | `2` | Min messages from a sender to suggest a rule |
| `--output` | stdout | Write accepted rules to a file |

### Example

```sh
# Analyze last 1000 messages, save accepted rules to a file
inbox_analyzer.py --config ~/mmuxer/config.yaml --limit 1000 --output new_rules.yaml
```
