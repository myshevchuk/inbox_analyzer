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
