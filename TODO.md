# Bugs

1. Folder name incorrectly includes a `.` stop when the "Service" token is a domain part (e.g. `Online.Wayfair.de` instead of `Online.Wayfair`). Likely comes from `from_display` rather than the anchor token. Also: expand the TLD list.

2. Despite being a TO-group, the rule is sometimes produced with a FROM-condition (e.g. `Online.Metasnake`). Not an isolated case.

3. `MessageGroup.to_addrs` appears to always be empty — possible bug in accumulation.

4. Coverage check uses loose substring matching rather than replicating mmuxer's actual matching logic, which can produce false positives and false negatives.

5. Batch fetch uses IMAP sequence numbers rather than UID FETCH, which is less robust.
