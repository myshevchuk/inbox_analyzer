# Bugs

1. Despite being a TO-group, the rule is sometimes produced with a FROM-condition (e.g. `Online.Metasnake`). Not an isolated case.

2. `MessageGroup.to_addrs` appears to always be empty — possible bug in accumulation.

3. Coverage check uses loose substring matching rather than replicating mmuxer's actual matching logic, which can produce false positives and false negatives.

4. Batch fetch uses IMAP sequence numbers rather than UID FETCH, which is less robust.
