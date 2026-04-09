"""aggregator_client URL normalization and small integration-style checks."""

from __future__ import annotations

import pytest

import aggregator_client as ac


def test_normalize_aggregator_base_url_adds_https():
    assert ac._normalize_aggregator_base_url("example.hf.space") == "https://example.hf.space"
    assert ac._normalize_aggregator_base_url("  example.hf.space/  ") == "https://example.hf.space"


def test_normalize_aggregator_base_url_preserves_explicit_scheme():
    assert ac._normalize_aggregator_base_url("http://127.0.0.1:7860") == "http://127.0.0.1:7860"
    assert ac._normalize_aggregator_base_url("https://x.hf.space/") == "https://x.hf.space"


def test_normalize_aggregator_base_url_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        ac._normalize_aggregator_base_url("  ")


def test_normalize_aggregator_base_url_fixes_owner_slash_space_hostname():
    wrong = "https://dev-the-dev91/instruction-fine-tuning-budget.hf.space"
    want = "https://dev-the-dev91-instruction-fine-tuning-budget.hf.space"
    assert ac._normalize_aggregator_base_url(wrong) == want
    assert (
        ac._normalize_aggregator_base_url(
            "dev-the-dev91/instruction-fine-tuning-budget.hf.space"
        )
        == want
    )


def test_normalize_aggregator_base_url_does_not_rewrite_real_hf_host_with_path():
    # Trailing slash only; no bogus merge
    assert ac._normalize_aggregator_base_url("https://ab-cd-ef.hf.space/") == (
        "https://ab-cd-ef.hf.space"
    )
