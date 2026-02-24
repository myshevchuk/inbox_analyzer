import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import inbox_analyzer as ia


def make_msg(**kwargs):
    defaults = {
        "from_addr": "sender@example.com",
        "from_display": "Sender Name",
        "to_addrs": [],
        "list_id": None,
        "subject": "Test Subject",
        "date": "Mon, 01 Jan 2024 00:00:00 +0000",
    }
    defaults.update(kwargs)
    return ia.MessageInfo(**defaults)


def make_config(rules=None, username="user@mydomain.com", server="imap.example.com", password="secret"):
    return {
        "settings": {"server": server, "username": username, "password": password},
        "rules": rules or [],
    }


def test_placeholder():
    assert True


def test_coverage_reason_uncovered():
    msg = make_msg(
        from_addr="unknown@example.com",
        to_addrs=["me@mydomain.com"],
        subject="Hello world",
    )
    covered = {"from": set(), "to": set(), "subject": set()}
    assert ia._coverage_reason(msg, covered) is None


def test_coverage_reason_from_exact():
    msg = make_msg(from_addr="newsletter@service.com")
    covered = {"from": {"newsletter@service.com"}, "to": set(), "subject": set()}
    result = ia._coverage_reason(msg, covered)
    assert result is not None
    assert result.startswith("FROM exact")


def test_coverage_reason_from_pattern():
    msg = make_msg(from_addr="updates@bigservice.com")
    covered = {"from": {"bigservice.com"}, "to": set(), "subject": set()}
    result = ia._coverage_reason(msg, covered)
    assert result is not None
    assert result.startswith("FROM pattern")


def test_coverage_reason_to_exact():
    msg = make_msg(
        from_addr="sender@other.com",
        to_addrs=["me+tag@mydomain.com"],
    )
    covered = {"from": set(), "to": {"me+tag@mydomain.com"}, "subject": set()}
    result = ia._coverage_reason(msg, covered)
    assert result is not None
    assert result.startswith("TO exact")


def test_coverage_reason_to_pattern():
    msg = make_msg(
        from_addr="sender@other.com",
        to_addrs=["me+newsletter@mydomain.com"],
    )
    covered = {"from": set(), "to": {"me+newsletter"}, "subject": set()}
    result = ia._coverage_reason(msg, covered)
    assert result is not None
    assert result.startswith("TO pattern")


def test_coverage_reason_subject():
    msg = make_msg(
        from_addr="sender@other.com",
        subject="Your weekly digest is ready",
    )
    covered = {"from": set(), "to": set(), "subject": {"weekly digest"}}
    result = ia._coverage_reason(msg, covered)
    assert result is not None
    assert result.startswith("SUBJECT")


def test_coverage_reason_from_takes_priority():
    msg = make_msg(
        from_addr="promo@shop.com",
        subject="special offer just for you",
    )
    covered = {
        "from": {"promo@shop.com"},
        "to": set(),
        "subject": {"special offer"},
    }
    result = ia._coverage_reason(msg, covered)
    assert result is not None
    assert result.startswith("FROM")


def test_extract_domain_tokens_simple():
    assert ia.ParsedAddress.from_addr("x@mail.google.com").domain_tokens == ["google"]


def test_extract_domain_tokens_co_uk():
    assert ia.ParsedAddress.from_addr("x@news.bbc.co.uk").domain_tokens == ["bbc"]


def test_extract_domain_tokens_subdomain():
    assert ia.ParsedAddress.from_addr("x@notifications.github.com").domain_tokens == ["github"]


def test_tokenize_basic():
    assert ia.tokenize("GitHub Notifications") == ["github"]


def test_tokenize_strips_stopwords():
    assert ia.tokenize("Hello from Spotify") == ["spotify"]


def test_tokenize_subject():
    assert ia.tokenize("Your invoice from Stripe") == ["invoice", "stripe"]


def test_find_strong_signals_overlap():
    result = ia.find_strong_signals(["spotify"], ["spotify", "music"], ["spotify", "playlist"])
    assert result == ["spotify"]


def test_find_strong_signals_no_overlap():
    result = ia.find_strong_signals(["apple"], ["banana"], ["cherry"])
    assert result == []


def test_stopwords_contains_expected():
    assert "newsletter" in ia.STOPWORDS
    assert "noreply" in ia.STOPWORDS


# ---------------------------------------------------------------------------
# extract_mydomain tests
# ---------------------------------------------------------------------------

def test_extract_mydomain_basic():
    config = make_config(username="user@mydomain.com")
    assert ia.extract_mydomain(config) == "mydomain.com"


def test_extract_mydomain_no_at():
    config = make_config(username="userwithoutatsign")
    assert ia.extract_mydomain(config) is None


def test_extract_mydomain_missing_settings():
    config = {"rules": []}
    assert ia.extract_mydomain(config) is None


# ---------------------------------------------------------------------------
# build_sender_index tests
# ---------------------------------------------------------------------------

def test_build_sender_index_basic():
    config = make_config(rules=[
        {"move_to": "Apps.Spotify", "condition": {"FROM": "noreply@spotify.com"}},
        {"move_to": "Apps.Music Production", "condition": {"FROM": "news@ableton.com"}},
    ])
    result = ia.build_sender_index(config)
    assert result == {
        "spotify": "Apps.Spotify",
        "music production": "Apps.Music Production",
    }


def test_build_sender_index_case_insensitive():
    config = make_config(rules=[
        {"move_to": "Apps.SPOTIFY", "condition": {"FROM": "noreply@spotify.com"}},
    ])
    result = ia.build_sender_index(config)
    assert "spotify" in result
    assert result["spotify"] == "Apps.SPOTIFY"


def test_build_sender_index_collision():
    config = make_config(rules=[
        {"move_to": "Apps.Alpha", "condition": {"FROM": "a@apps.com"}},
        {"move_to": "Admin.Alpha", "condition": {"FROM": "a@admin.com"}},
    ])
    result = ia.build_sender_index(config)
    # "Admin.Alpha" < "Apps.Alpha" alphabetically, so Admin.Alpha wins
    assert result["alpha"] == "Admin.Alpha"


def test_build_sender_index_no_dot():
    config = make_config(rules=[
        {"move_to": "Spotify", "condition": {"FROM": "noreply@spotify.com"}},
    ])
    result = ia.build_sender_index(config)
    assert result == {"spotify": "Spotify"}


def test_build_sender_index_ignores_delete_rules():
    config = make_config(rules=[
        {"actions": ["delete"], "condition": {"FROM": "spam@junk.com"}},
    ])
    result = ia.build_sender_index(config)
    assert result == {}


# ---------------------------------------------------------------------------
# match_tokens_to_rule tests
# ---------------------------------------------------------------------------

def test_match_tokens_no_match():
    sender_index = {"spotify": "Apps.Spotify", "github": "Dev.GitHub"}
    result = ia.match_tokens_to_rule(["dropbox", "cloud"], sender_index)
    assert result is None


def test_match_tokens_single_match():
    sender_index = {"spotify": "Apps.Spotify", "github": "Dev.GitHub"}
    result = ia.match_tokens_to_rule(["spotify", "music"], sender_index)
    assert result == "Apps.Spotify"


def test_match_tokens_best_score_wins():
    # "music production" splits into ["music", "production"], matching both tokens
    # "spotify" matches only one token
    sender_index = {
        "spotify": "Apps.Spotify",
        "music production": "Apps.Music Production",
    }
    result = ia.match_tokens_to_rule(["music", "production"], sender_index)
    assert result == "Apps.Music Production"


def test_match_tokens_tie_broken_by_to_alias():
    # Both "alpha" keys score 1 (token "alpha" appears in each)
    # to_alias "Apps" matches category of "Apps.Alpha"
    sender_index = {
        "alpha beta": "Apps.Alpha",
        "alpha gamma": "Admin.Alpha",
    }
    result = ia.match_tokens_to_rule(["alpha"], sender_index, to_alias="Apps")
    assert result == "Apps.Alpha"


def test_match_tokens_tie_broken_alphabetically():
    # Both score 1; no to_alias — alphabetical on folder path
    sender_index = {
        "alpha beta": "Zzz.Alpha",
        "alpha gamma": "Aaa.Alpha",
    }
    result = ia.match_tokens_to_rule(["alpha"], sender_index)
    assert result == "Aaa.Alpha"


def test_match_tokens_empty_tokens():
    sender_index = {"spotify": "Apps.Spotify"}
    result = ia.match_tokens_to_rule([], sender_index)
    assert result is None


# ---------------------------------------------------------------------------
# classify_and_group_emails tests
# ---------------------------------------------------------------------------

def test_classify_subaddress_group():
    msg = make_msg(
        from_addr="updates@spotify.com",
        from_display="Spotify",
        to_addrs=["apps+spotify@mydomain.com"],
        subject="Your Spotify playlist",
    )
    config = make_config()
    result = ia.classify_and_group_emails([msg], config, mydomain="mydomain.com")
    assert len(result) == 1
    group = result[0]
    assert group.anchor_type == "TO"
    assert group.group_key == "TO:apps+spotify"


def test_classify_from_group():
    from_addr = "updates@example.com"
    msg = make_msg(
        from_addr=from_addr,
        from_display="Example Service",
        to_addrs=["me@mydomain.com"],
        subject="Hello",
        list_id=None,
    )
    config = make_config()
    result = ia.classify_and_group_emails([msg], config, mydomain="mydomain.com")
    assert len(result) == 1
    group = result[0]
    assert group.anchor_type == "FROM"
    assert group.group_key == f"FROM:{from_addr}"


def test_classify_list_group():
    msg = make_msg(
        from_addr="sender@list.example.com",
        from_display="List Sender",
        to_addrs=["me@mydomain.com"],
        subject="List message",
        list_id="<list.example.com>",
    )
    config = make_config()
    result = ia.classify_and_group_emails([msg], config, mydomain="mydomain.com")
    assert len(result) == 1
    group = result[0]
    assert group.anchor_type == "LIST"
    assert group.group_key.startswith("LIST:")


def test_classify_suggested_destination():
    msg = make_msg(
        from_addr="sender@spotify.com",
        from_display="Spotify",
        to_addrs=["me@mydomain.com"],
        subject="Your Spotify update",
        list_id=None,
    )
    config = make_config(rules=[
        {"move_to": "Apps.Spotify", "condition": {"FROM": "noreply@spotify.com"}},
    ])
    result = ia.classify_and_group_emails([msg], config, mydomain="mydomain.com")
    assert len(result) == 1
    group = result[0]
    assert group.suggested_destination == "Apps.Spotify"


def test_classify_cross_group_linking():
    msg_to = make_msg(
        from_addr="updates@spotify.com",
        from_display="Spotify",
        to_addrs=["apps+spotify@mydomain.com"],
        subject="Your Spotify playlist",
    )
    msg_from = make_msg(
        from_addr="updates@spotify.com",
        from_display="Spotify Updates",
        to_addrs=["me@mydomain.com"],
        subject="Spotify news",
        list_id=None,
    )
    config = make_config()
    result = ia.classify_and_group_emails([msg_to, msg_from], config, mydomain="mydomain.com")
    to_group = next(g for g in result if g.anchor_type == "TO")
    from_group = next(g for g in result if g.anchor_type == "FROM")
    assert from_group.group_key in to_group.related_group_keys or to_group.group_key in from_group.related_group_keys


def test_classify_groups_accumulated():
    msg1 = make_msg(
        from_addr="news@example.com",
        from_display="Example News",
        to_addrs=["me@mydomain.com"],
        subject="First message",
    )
    msg2 = make_msg(
        from_addr="news@example.com",
        from_display="Example News",
        to_addrs=["me@mydomain.com"],
        subject="Second message",
    )
    config = make_config()
    result = ia.classify_and_group_emails([msg1, msg2], config, mydomain="mydomain.com")
    assert len(result) == 1
    assert result[0].count == 2


# ---------------------------------------------------------------------------
# suggest_folder_name tests
# ---------------------------------------------------------------------------

def test_suggest_folder_name_uses_suggested_destination():
    group = ia.MessageGroup(
        from_addr="noreply@spotify.com",
        from_display="Spotify",
        count=1,
        sample_subjects=[],
        suggested_destination="Apps.Spotify",
    )
    assert ia.suggest_folder_name(group) == "Apps.Spotify"


def test_suggest_folder_name_fallback_display():
    group = ia.MessageGroup(
        from_addr="noreply@github.com",
        from_display="GitHub",
        count=1,
        sample_subjects=[],
        suggested_destination=None,
    )
    assert ia.suggest_folder_name(group) == "GitHub"


def test_suggest_folder_name_fallback_domain():
    group = ia.MessageGroup(
        from_addr="noreply@service.com",
        from_display="",
        count=1,
        sample_subjects=[],
        suggested_destination=None,
    )
    assert ia.suggest_folder_name(group) == "Service"


# ---------------------------------------------------------------------------
# suggest_rule tests
# ---------------------------------------------------------------------------

def test_suggest_rule_uses_suggested_destination():
    group = ia.MessageGroup(
        from_addr="noreply@spotify.com",
        from_display="Spotify",
        count=1,
        sample_subjects=[],
        from_addrs={"noreply@spotify.com"},
        suggested_destination="Apps.Spotify",
    )
    rule = ia.suggest_rule(group)
    assert "move_to: Apps.Spotify" in rule


def test_suggest_rule_to_anchor_type():
    group = ia.MessageGroup(
        from_addr="updates@spotify.com",
        from_display="Spotify",
        count=1,
        sample_subjects=[],
        from_addrs={"updates@spotify.com"},
        to_addrs={"apps+spotify@mydomain.com"},
        anchor_type="TO",
    )
    rule = ia.suggest_rule(group)
    assert "TO: apps+spotify@mydomain.com" in rule


def test_suggest_rule_multi_sender():
    group = ia.MessageGroup(
        from_addr="a@x.com",
        from_display="X Service",
        count=2,
        sample_subjects=[],
        from_addrs={"a@x.com", "b@x.com"},
    )
    rule = ia.suggest_rule(group)
    assert "ANY:" in rule
    assert "FROM: a@x.com" in rule
    assert "FROM: b@x.com" in rule


# ---------------------------------------------------------------------------
# Integration test: end-to-end pipeline without IMAP
# ---------------------------------------------------------------------------

def test_suggest_folder_name_to_group_no_destination():
    """TO group without suggested_destination uses prefix.tag from subaddress."""
    group = ia.MessageGroup(
        from_addr="noreply@unrelated.com",
        from_display="Some Sender",
        count=1,
        sample_subjects=[],
        group_key="TO:apps+spotify",
        anchor_type="TO",
    )
    assert ia.suggest_folder_name(group) == "Apps.Spotify"


def test_to_group_tag_drives_anchor_tokens():
    """Tag part of subaddress (apps+spotify → 'spotify') must appear in anchor_tokens."""
    msg = make_msg(
        from_addr="news@unrelated-domain.com",
        from_display="Some Newsletter",
        to_addrs=["apps+spotify@mydomain.com"],
    )
    config = make_config()
    result = ia.classify_and_group_emails([msg], config, "mydomain.com")
    assert len(result) == 1
    group = result[0]
    assert group.anchor_type == "TO"
    assert "spotify" in group.anchor_tokens


def test_to_group_tag_matches_suggested_destination():
    """Tag token should match sender_index even when sender domain is unrelated."""
    msg = make_msg(
        from_addr="noreply@unrelated-domain.com",
        from_display="Unrelated Sender",
        to_addrs=["apps+spotify@mydomain.com"],
    )
    config = make_config(rules=[
        {"move_to": "Apps.Spotify", "condition": {"FROM": "noreply@spotify.com"}},
    ])
    result = ia.classify_and_group_emails([msg], config, "mydomain.com")
    assert len(result) == 1
    assert result[0].suggested_destination == "Apps.Spotify"


def test_pipeline_end_to_end():
    msgs = [
        make_msg(
            from_addr="news@spotify.com",
            from_display="Spotify",
            to_addrs=["apps+spotify@mydomain.com"],
            subject="Your Spotify update",
        ),
        make_msg(
            from_addr="updates@spotify.com",
            from_display="Spotify Updates",
            to_addrs=["me@mydomain.com"],
            subject="Spotify news",
        ),
    ]
    config = make_config(rules=[
        {"move_to": "Apps.Spotify", "condition": {"FROM": "something@spotify.com"}},
    ])
    result = ia.classify_and_group_emails(msgs, config, "mydomain.com")
    assert len(result) >= 1
    for group in result:
        assert group.group_key != ""


# ---------------------------------------------------------------------------
# to_alias tests
# ---------------------------------------------------------------------------

def test_to_alias_extracted_for_from_group():
    """FROM group message with to_addrs=["forma@mydomain.com"], primary_local="user" → to_alias="forma"."""
    msg = make_msg(
        from_addr="bank@somebank.com",
        from_display="Bank",
        to_addrs=["forma@mydomain.com"],
        subject="Your statement",
    )
    config = make_config()
    result = ia.classify_and_group_emails([msg], config, mydomain="mydomain.com", primary_local="user")
    assert len(result) == 1
    group = result[0]
    assert group.to_alias == "forma"


def test_to_alias_not_set_for_primary_inbox():
    """FROM group message with to_addrs=["user@mydomain.com"], primary_local="user" → to_alias=""."""
    msg = make_msg(
        from_addr="bank@somebank.com",
        from_display="Bank",
        to_addrs=["user@mydomain.com"],
        subject="Your statement",
    )
    config = make_config()
    result = ia.classify_and_group_emails([msg], config, mydomain="mydomain.com", primary_local="user")
    assert len(result) == 1
    group = result[0]
    assert group.to_alias == ""


def test_to_alias_from_subaddress_prefix():
    """TO group message with to_addrs=["apps+spotify@mydomain.com"] → to_alias is present (set to "apps")."""
    msg = make_msg(
        from_addr="noreply@spotify.com",
        from_display="Spotify",
        to_addrs=["apps+spotify@mydomain.com"],
        subject="Your playlist",
    )
    config = make_config()
    result = ia.classify_and_group_emails([msg], config, mydomain="mydomain.com", primary_local="user")
    assert len(result) == 1
    group = result[0]
    assert group.anchor_type == "TO"
    assert group.to_alias == "apps"


def test_suggest_folder_name_with_to_alias():
    """FROM group with to_alias="forma", from_display="Bank Name" → returns "Forma.Bank Name"."""
    group = ia.MessageGroup(
        from_addr="info@somebank.com",
        from_display="Bank Name",
        count=1,
        sample_subjects=[],
        anchor_type="FROM",
        group_key="FROM:info@somebank.com",
        to_alias="forma",
    )
    assert ia.suggest_folder_name(group) == "Forma.Bank Name"


def test_suggest_folder_name_no_category_unchanged():
    """FROM group with to_alias="", from_display="Bank Name" → returns "Bank Name" unchanged."""
    group = ia.MessageGroup(
        from_addr="info@somebank.com",
        from_display="Bank Name",
        count=1,
        sample_subjects=[],
        anchor_type="FROM",
        group_key="FROM:info@somebank.com",
        to_alias="",
    )
    assert ia.suggest_folder_name(group) == "Bank Name"


def test_to_alias_ignored_when_no_mydomain():
    """Message with to_addrs=["forma@mydomain.com"], mydomain=None → to_alias=""."""
    msg = make_msg(
        from_addr="bank@somebank.com",
        from_display="Bank",
        to_addrs=["forma@mydomain.com"],
        subject="Your statement",
    )
    config = make_config()
    result = ia.classify_and_group_emails([msg], config, mydomain=None, primary_local="user")
    assert len(result) == 1
    group = result[0]
    assert group.to_alias == ""
