"""Namespace package providing hooks and test counters."""
from __future__ import annotations

_mock_counters = {"ocr": 0, "md": 0}


def reset_mock_counters() -> None:
    for k in _mock_counters:
        _mock_counters[k] = 0


def get_mock_counters() -> dict:
    return dict(_mock_counters)

# place holders for optional monkeypatched functions
# e.g., llm_embed, chat_json etc.
