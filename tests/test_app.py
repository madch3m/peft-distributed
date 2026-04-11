"""
Runtime checks for the aggregator Space (FastAPI + Gradio helpers).

Uses empty HF credentials so FedAvg skips Hub I/O (no network required).
"""

from __future__ import annotations

import json
import time
from unittest import mock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """Fresh TestClient with isolated config/state (no Hub calls)."""
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("MODEL_REPO_ID", "")
    monkeypatch.setenv("NODE_SECRET", "test_secret_roundtrip")
    monkeypatch.setenv("RATE_LIMIT_SUBMIT_MAX", "10000")
    monkeypatch.setenv("RATE_LIMIT_SUBMIT_WINDOW_SEC", "60")

    import importlib

    import app as app_module

    importlib.reload(app_module)
    rb = getattr(app_module, "_submit_rate_buckets", None)
    if rb is not None:
        rb.clear()

    # Hard reset in-memory state (reload re-executes module but keeps clarity)
    app_module.state["current_round"] = 1
    app_module.state["submitted_nodes"] = []
    app_module.state["history"] = []
    app_module.state["last_update"] = None
    app_module.state["node_metrics"] = {}
    app_module.state["activity_log"] = []
    app_module.state["merging"] = False
    app_module.state["merge_error"] = None

    with TestClient(app_module.app) as c:
        yield c, app_module


def test_health_ok(client):
    tc, _ = client
    r = tc.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_status_ok(client):
    tc, _ = client
    r = tc.get("/status")
    assert r.status_code == 200
    data = r.json()
    assert data["current_round"] == 1
    assert data["submitted_nodes"] == []
    assert "node_a" in data["expected_nodes"]


def test_submit_rate_limit(client, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("MODEL_REPO_ID", "")
    monkeypatch.setenv("NODE_SECRET", "rl_secret")
    monkeypatch.setenv("RATE_LIMIT_SUBMIT_MAX", "4")
    monkeypatch.setenv("RATE_LIMIT_SUBMIT_WINDOW_SEC", "300")

    import importlib

    import app as app_module

    importlib.reload(app_module)
    rb = getattr(app_module, "_submit_rate_buckets", None)
    if rb is not None:
        rb.clear()
    app_module.state["current_round"] = 1
    app_module.state["submitted_nodes"] = []
    app_module.state["node_metrics"] = {}

    with TestClient(app_module.app) as tc:
        secret = "rl_secret"
        body = {"node_id": "node_a", "secret_key": secret}
        for i in range(4):
            r = tc.post("/submit", json=body)
            assert r.status_code == 200, r.text
        r5 = tc.post("/submit", json=body)
        assert r5.status_code == 429
        assert "Too many" in r5.json().get("detail", "")


def test_submit_invalid_secret(client):
    tc, m = client
    r = tc.post(
        "/submit",
        json={"node_id": "node_a", "secret_key": "wrong"},
    )
    assert r.status_code == 401


def test_submit_round_num_mismatch(client):
    tc, m = client
    secret = m.CONFIG["node_secret"]
    r = tc.post(
        "/submit",
        json={
            "node_id": "node_a",
            "secret_key": secret,
            "round_num": 99,
        },
    )
    assert r.status_code == 409
    assert "current_round" in r.json()["detail"]


def test_submit_avg_loss_validation(client):
    tc, m = client
    secret = m.CONFIG["node_secret"]
    r = tc.post(
        "/submit",
        json={
            "node_id": "node_a",
            "secret_key": secret,
            "avg_loss": 2.0e7,
        },
    )
    assert r.status_code == 422


def test_submit_rejects_unknown_json_fields(client):
    tc, m = client
    secret = m.CONFIG["node_secret"]
    r = tc.post(
        "/submit",
        json={
            "node_id": "node_a",
            "secret_key": secret,
            "extra_field": 1,
        },
    )
    assert r.status_code == 422


def test_submit_unknown_node(client):
    tc, m = client
    r = tc.post(
        "/submit",
        json={"node_id": "node_x", "secret_key": m.CONFIG["node_secret"]},
    )
    assert r.status_code == 400


def _wait_for_merge(app_module, timeout=5):
    """Wait until the background merge thread completes."""
    deadline = time.monotonic() + timeout
    while app_module.state["merging"] and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not app_module.state["merging"], "merge did not complete in time"


def test_full_round_merge_skipped_no_hf(client):
    tc, m = client
    secret = m.CONFIG["node_secret"]
    losses = {"node_a": 1.5, "node_b": 1.6, "node_c": 1.55}
    steps = {"node_a": 100, "node_b": 101, "node_c": 99}
    for nid in ("node_a", "node_b", "node_c"):
        r = tc.post(
            "/submit",
            json={
                "node_id": nid,
                "secret_key": secret,
                "avg_loss": losses[nid],
                "steps_completed": steps[nid],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        if nid != "node_c":
            assert body["status"] == "submitted"
        else:
            # Last node triggers async merge — returns "merging"
            assert body["status"] == "merging"

    # Wait for the background merge thread to finish
    _wait_for_merge(m)

    st = tc.get("/status").json()
    assert st["current_round"] == 2
    assert st["submitted_nodes"] == []
    assert len(m.state["history"]) == 1

    last = m.state["history"][-1]
    for nid in ("node_a", "node_b", "node_c"):
        assert last["node_losses"][nid] == losses[nid]
        assert last["node_steps"][nid] == steps[nid]
    assert m.state["node_metrics"] == {}


def test_reset(client):
    tc, m = client
    secret = m.CONFIG["node_secret"]
    tc.post("/submit", json={"node_id": "node_a", "secret_key": secret})
    r = tc.post("/reset", json={"secret_key": secret})
    assert r.status_code == 200
    assert r.json()["current_round"] == 1
    assert m.state["submitted_nodes"] == []


def test_reset_requires_admin_when_admin_secret_set(client, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("MODEL_REPO_ID", "")
    monkeypatch.setenv("NODE_SECRET", "node_only_secret")
    monkeypatch.setenv("ADMIN_SECRET", "admin_only_secret")
    monkeypatch.setenv("RATE_LIMIT_SUBMIT_MAX", "10000")
    monkeypatch.setenv("RATE_LIMIT_SUBMIT_WINDOW_SEC", "60")

    import importlib

    import app as app_module

    importlib.reload(app_module)
    rb = getattr(app_module, "_submit_rate_buckets", None)
    if rb is not None:
        rb.clear()
    app_module.state["current_round"] = 1
    app_module.state["submitted_nodes"] = []
    app_module.state["node_metrics"] = {}

    with TestClient(app_module.app) as tc:
        tc.post(
            "/submit",
            json={"node_id": "node_a", "secret_key": "node_only_secret"},
        )
        bad = tc.post("/reset", json={"secret_key": "node_only_secret"})
        assert bad.status_code == 401
        good = tc.post("/reset", json={"secret_key": "admin_only_secret"})
        assert good.status_code == 200


def test_status_requires_header_when_status_read_secret_set(client, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("MODEL_REPO_ID", "")
    monkeypatch.setenv("NODE_SECRET", "test_secret_roundtrip")
    monkeypatch.setenv("STATUS_READ_SECRET", "peek_abcd")
    monkeypatch.setenv("RATE_LIMIT_SUBMIT_MAX", "10000")
    monkeypatch.setenv("RATE_LIMIT_SUBMIT_WINDOW_SEC", "60")

    import importlib

    import app as app_module

    importlib.reload(app_module)
    rb = getattr(app_module, "_submit_rate_buckets", None)
    if rb is not None:
        rb.clear()
    app_module.state["current_round"] = 1
    app_module.state["submitted_nodes"] = []

    with TestClient(app_module.app) as tc:
        denied = tc.get("/status")
        assert denied.status_code == 401
        ok = tc.get("/status", headers={"X-Status-Secret": "peek_abcd"})
        assert ok.status_code == 200
        assert ok.json()["current_round"] == 1


def test_gradio_markdown_helpers_no_crash(client):
    """Dashboard builders used by Gradio refresh must not raise."""
    import matplotlib.pyplot as plt

    tc, m = client
    m._round_progress_md()
    m._node_cards_md()
    m._loss_history_md()
    assert m._convergence_figure() is None
    assert m._steps_bar_figure() is None
    m._merged_adapters_md()
    m._activity_log_md()

    secret = m.CONFIG["node_secret"]
    for nid in ("node_a", "node_b", "node_c"):
        tc.post("/submit", json={"node_id": nid, "secret_key": secret})

    _wait_for_merge(m)

    m._loss_history_md()
    m._merged_adapters_md()
    m._activity_log_md()

    fig_c = m._convergence_figure()
    if fig_c is not None:
        plt.close(fig_c)
    fig_s = m._steps_bar_figure()
    if fig_s is not None:
        plt.close(fig_s)


def test_restore_state_from_hub(monkeypatch, tmp_path):
    """Simulate a Space restart that restores state from Hub."""
    monkeypatch.setenv("HF_TOKEN", "fake_token")
    monkeypatch.setenv("MODEL_REPO_ID", "fake/repo")
    monkeypatch.setenv("NODE_SECRET", "test_secret_roundtrip")
    monkeypatch.setenv("RATE_LIMIT_SUBMIT_MAX", "10000")
    monkeypatch.setenv("RATE_LIMIT_SUBMIT_WINDOW_SEC", "60")

    # Write a fake aggregator_state.json that hf_hub_download will "return"
    saved_state = {
        "current_round": 4,
        "last_merged_round": 3,
        "history": [
            {"round": 1, "completed_at": "t1", "merge_result": "ok",
             "node_losses": {"node_a": 1.5, "node_b": 1.6, "node_c": 1.55},
             "node_steps": {"node_a": 100, "node_b": 100, "node_c": 100}},
            {"round": 2, "completed_at": "t2", "merge_result": "ok",
             "node_losses": {"node_a": 1.3, "node_b": 1.4, "node_c": 1.35},
             "node_steps": {"node_a": 100, "node_b": 100, "node_c": 100}},
            {"round": 3, "completed_at": "t3", "merge_result": "ok",
             "node_losses": {"node_a": 1.1, "node_b": 1.2, "node_c": 1.15},
             "node_steps": {"node_a": 100, "node_b": 100, "node_c": 100}},
        ],
        "timestamp": "2026-04-11T00:00:00Z",
    }
    state_file = tmp_path / "aggregator_state.json"
    state_file.write_text(json.dumps(saved_state))

    def fake_hf_hub_download(repo_id, filename, token):
        if filename == "aggregator_state.json":
            return str(state_file)
        raise FileNotFoundError(filename)

    import importlib
    import app as app_module

    # Patch hf_hub_download before reload so _restore_state_from_hub uses it
    with mock.patch.object(app_module, "hf_hub_download", side_effect=fake_hf_hub_download):
        importlib.reload(app_module)
        # Re-patch after reload (reload re-imports the name)
        app_module.hf_hub_download = fake_hf_hub_download
        app_module._restore_state_from_hub()

    assert app_module.state["current_round"] == 4
    assert len(app_module.state["history"]) == 3
    assert app_module.state["submitted_nodes"] == []
    assert app_module.state["merging"] is False

    # Verify the dashboard still works with restored history
    with TestClient(app_module.app) as tc:
        st = tc.get("/status").json()
        assert st["current_round"] == 4


def test_persist_state_no_op_without_credentials(client):
    """_persist_state_to_hub should silently no-op when HF creds are empty."""
    _, m = client
    # Should not raise
    m._persist_state_to_hub()


def test_openapi_lists_core_routes(client):
    """FastAPI JSON routes (used by nodes); Gradio HTML at / is not exercised here
    (Starlette TestClient can hit a Gradio/Jinja edge case unrelated to Space runtime).
    """
    tc, _ = client
    r = tc.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json().get("paths", {})
    assert "/health" in paths
    assert "/status" in paths
    assert "/submit" in paths
    assert "/reset" in paths
