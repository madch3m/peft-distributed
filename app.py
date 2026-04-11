"""
app.py — Aggregator Space for Distributed PEFT Fine-Tuning.

Gradio dashboard + FastAPI endpoints:
    /submit  — Node reports round completion
    /status  — Current aggregation state (JSON)
    /reset   — Reset to round 1

FedAvg pipeline: when all 3 nodes submit, averages adapter states on HF Hub.

Optional env: ADMIN_SECRET (reset-only), STATUS_READ_SECRET (GET /status via X-Status-Secret).
"""

import logging
import math
import os
import json
import threading
import time
import datetime

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, field_validator
from starlette.responses import JSONResponse
from huggingface_hub import HfApi, hf_hub_download, upload_file
from safetensors.torch import load_file, save_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG = {
    "expected_nodes": ["node_a", "node_b", "node_c"],
    "node_secret": os.environ.get("NODE_SECRET", "local_test_secret"),
    "admin_secret": os.environ.get("ADMIN_SECRET", "").strip(),
    "status_read_secret": os.environ.get("STATUS_READ_SECRET", "").strip(),
    "hf_token": os.environ.get("HF_TOKEN", ""),
    "model_repo_id": os.environ.get("MODEL_REPO_ID", ""),
    "rate_limit_submit_max": int(os.environ.get("RATE_LIMIT_SUBMIT_MAX", "120")),
    "rate_limit_submit_window_sec": int(
        os.environ.get("RATE_LIMIT_SUBMIT_WINDOW_SEC", "60")
    ),
    "node_labels": {
        "node_a": "Node A (Layers 0–10)",
        "node_b": "Node B (Layers 11–21)",
        "node_c": "Node C (Layers 22–31)",
    },
}


def _agg_logger() -> logging.Logger:
    lg = logging.getLogger("peft.aggregator")
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)s [peft_agg] %(message)s"))
        lg.addHandler(h)
    lg.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    return lg


agg_log = _agg_logger()


def _effective_reset_secret() -> str:
    """POST /reset accepts ADMIN_SECRET when set; otherwise NODE_SECRET."""
    if CONFIG["admin_secret"]:
        return CONFIG["admin_secret"]
    return CONFIG["node_secret"]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

state = {
    "current_round": 1,
    "submitted_nodes": [],
    "history": [],
    "last_update": None,
    "node_metrics": {},       # {node_id: {loss, step, timestamp, ...}}
    "activity_log": [],       # recent events for the activity feed
    "merging": False,         # True while FedAvg is running in background
    "merge_error": None,      # set if background merge failed
}

_state_lock = threading.Lock()


def _persist_state_to_hub() -> None:
    """Upload aggregator state to Hub so a Space restart can resume.

    Called after round completion and reset.  Runs inside the background
    merge thread (or the reset endpoint), so it won't block node requests.
    """
    repo = CONFIG["model_repo_id"]
    token = CONFIG["hf_token"]
    if not repo or not token:
        return

    payload = {
        "current_round": state["current_round"],
        "last_merged_round": state["current_round"] - 1,
        "history": state["history"],
        "timestamp": _timestamp(),
    }
    path = "/tmp/aggregator_state.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    try:
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo="aggregator_state.json",
            repo_id=repo,
            token=token,
            commit_message=f"Aggregator state — round {state['current_round']}",
        )
        agg_log.info("persisted aggregator state to Hub (round %s)", state["current_round"])
    except Exception as e:
        agg_log.warning("failed to persist aggregator state: %s", e)


def _restore_state_from_hub() -> None:
    """On startup, download aggregator_state.json (or training_state.json)
    from Hub and restore current_round + history.

    Falls back to training_state.json (written by fedavg_merge) if the
    newer aggregator_state.json does not exist yet.
    """
    repo = CONFIG["model_repo_id"]
    token = CONFIG["hf_token"]
    if not repo or not token:
        agg_log.info("no HF credentials — starting from round 1")
        return

    # Try aggregator_state.json first (has history), then training_state.json
    for filename in ("aggregator_state.json", "training_state.json"):
        try:
            path = hf_hub_download(repo_id=repo, filename=filename, token=token)
            with open(path) as f:
                saved = json.load(f)

            restored_round = saved.get("current_round", 1)
            state["current_round"] = restored_round
            state["history"] = saved.get("history", [])
            state["last_update"] = saved.get("timestamp")
            state["submitted_nodes"] = []
            state["node_metrics"] = {}
            state["merging"] = False
            state["merge_error"] = None

            _log_activity(f"State restored from Hub ({filename}) — resuming at round {restored_round}")
            agg_log.info(
                "restored state from %s: round=%s history_len=%s",
                filename,
                restored_round,
                len(state["history"]),
            )
            return
        except Exception:
            continue

    agg_log.info("no saved state on Hub — starting from round 1")


# Restore on import (runs when the Space boots)
_restore_state_from_hub()

# Per-client timestamps (monotonic) for POST /submit rate limiting
_submit_rate_buckets: dict[str, list[float]] = {}


def _submit_rate_allow(client_key: str) -> bool:
    """Return True if the client may POST /submit (sliding window)."""
    now = time.monotonic()
    window = max(1, CONFIG["rate_limit_submit_window_sec"])
    cap = max(1, CONFIG["rate_limit_submit_max"])
    bucket = _submit_rate_buckets.setdefault(client_key, [])
    cutoff = now - window
    while bucket and bucket[0] < cutoff:
        bucket.pop(0)
    if len(bucket) >= cap:
        return False
    bucket.append(now)
    return True


def _client_key_for_rate_limit(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip() or "unknown"
    if request.client:
        return request.client.host
    return "unknown"


def _timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _log_activity(message: str):
    """Append a timestamped entry to the activity log (keep last 50)."""
    state["activity_log"].append({
        "time": _timestamp(),
        "message": message,
    })
    state["activity_log"] = state["activity_log"][-50:]


# ---------------------------------------------------------------------------
# FedAvg
# ---------------------------------------------------------------------------

def fedavg_merge() -> tuple[str, bool]:
    """Download adapter states from all nodes, average, upload merged checkpoint.

    Returns:
        (message, success). success=False means the round should not advance.
    """
    api = HfApi(token=CONFIG["hf_token"])
    repo = CONFIG["model_repo_id"]
    rnd = state["current_round"]

    if not repo or not CONFIG["hf_token"]:
        return "Skipped merge — MODEL_REPO_ID or HF_TOKEN not set.", True

    adapter_states = []
    for node_id in CONFIG["expected_nodes"]:
        try:
            path = hf_hub_download(
                repo_id=repo,
                filename=f"{node_id}/adapter_model.safetensors",
                token=CONFIG["hf_token"],
            )
            adapter_states.append(load_file(path))
        except Exception as e:
            return f"FedAvg failed — could not load {node_id}: {e}", False

    ref_keys = set(adapter_states[0].keys())
    for i, node_id in enumerate(CONFIG["expected_nodes"]):
        keys_i = set(adapter_states[i].keys())
        if keys_i != ref_keys:
            missing = ref_keys - keys_i
            extra = keys_i - ref_keys
            return (
                f"FedAvg failed — {node_id} tensor keys mismatch "
                f"(missing={sorted(missing)[:5]!s} extra={sorted(extra)[:5]!s})",
                False,
            )

    merged_path = "/tmp/merged_adapter_model.safetensors"
    try:
        merged = {}
        for key in ref_keys:
            stacked = [s[key].float() for s in adapter_states]
            merged[key] = sum(stacked) / len(stacked)
        save_file(merged, merged_path)
    except Exception as e:
        return f"FedAvg failed — tensor merge/save: {e}", False

    try:
        api.upload_file(
            path_or_fileobj=merged_path,
            path_in_repo=f"merged/round_{rnd}/adapter_model.safetensors",
            repo_id=repo,
            token=CONFIG["hf_token"],
            commit_message=f"FedAvg merge — round {rnd}",
        )

        training_state = {
            "current_round": rnd + 1,
            "last_merged_round": rnd,
            "timestamp": _timestamp(),
        }
        state_path = "/tmp/training_state.json"
        with open(state_path, "w") as f:
            json.dump(training_state, f, indent=2)

        api.upload_file(
            path_or_fileobj=state_path,
            path_in_repo="training_state.json",
            repo_id=repo,
            token=CONFIG["hf_token"],
            commit_message=f"Advance to round {rnd + 1}",
        )
    except Exception as e:
        return f"FedAvg failed — Hub upload: {e}", False

    return (
        f"FedAvg complete for round {rnd}. Advanced to round {rnd + 1}.",
        True,
    )


def _background_merge(
    merge_round: int,
    round_metrics: dict,
    round_steps: dict,
) -> None:
    """Run FedAvg in a background thread, then advance the round or record failure."""
    try:
        merge_result, merge_ok = fedavg_merge()
    except Exception as exc:
        merge_result, merge_ok = f"FedAvg exception: {exc}", False

    with _state_lock:
        if merge_ok:
            state["history"].append({
                "round": merge_round,
                "completed_at": _timestamp(),
                "merge_result": merge_result,
                "node_losses": round_metrics,
                "node_steps": round_steps,
            })
            _log_activity(f"Round {merge_round} FedAvg complete")
            agg_log.info(
                "fedavg_complete round=%s new_round=%s",
                merge_round,
                merge_round + 1,
            )
            state["current_round"] = merge_round + 1
            state["submitted_nodes"] = []
            state["node_metrics"] = {}
        else:
            tail = merge_result if len(merge_result) <= 400 else merge_result[:400] + "..."
            agg_log.warning("fedavg_failed round=%s detail=%s", merge_round, tail)
            _log_activity(f"FedAvg failed (round {merge_round}): {merge_result}")
            state["merge_error"] = merge_result
            state["submitted_nodes"] = []

        state["merging"] = False

    # Persist outside the lock (Hub I/O is slow)
    if merge_ok:
        _persist_state_to_hub()


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------

app = FastAPI(title="PEFT Aggregator")


@app.middleware("http")
async def limit_submit_rate(request: Request, call_next):
    if request.method != "POST":
        return await call_next(request)
    path = request.url.path.rstrip("/") or "/"
    if not path.endswith("/submit"):
        return await call_next(request)
    key = _client_key_for_rate_limit(request)
    if not _submit_rate_allow(key):
        agg_log.warning("rate_limit exceeded client_key=%s", key[:128])
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many submit requests; try again later."},
        )
    return await call_next(request)


@app.get("/health")
def health():
    """Liveness probe — no Hub calls, no mutation of training state."""
    return {"ok": True}


class SubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    secret_key: str
    round_num: int | None = None
    avg_loss: float | None = None
    steps_completed: int | None = None

    @field_validator("avg_loss")
    @classmethod
    def avg_loss_finite(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if not math.isfinite(v) or abs(v) > 1.0e7:
            raise ValueError("avg_loss must be finite and |avg_loss| <= 1e7")
        return v

    @field_validator("steps_completed")
    @classmethod
    def steps_in_range(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if v < 0 or v > 10**9:
            raise ValueError("steps_completed must be between 0 and 1e9 inclusive")
        return v


class ResetRequest(BaseModel):
    secret_key: str


@app.get("/status")
def get_status(request: Request):
    sr = CONFIG["status_read_secret"]
    if sr:
        hdr = request.headers.get("x-status-secret", "")
        if hdr != sr:
            agg_log.warning("status rejected: invalid or missing X-Status-Secret")
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing X-Status-Secret header",
            )
    return {
        "current_round": state["current_round"],
        "submitted_nodes": state["submitted_nodes"],
        "expected_nodes": CONFIG["expected_nodes"],
        "remaining": [
            n for n in CONFIG["expected_nodes"]
            if n not in state["submitted_nodes"]
        ],
        "last_update": state["last_update"],
        "merging": state["merging"],
        "merge_error": state["merge_error"],
    }


@app.post("/submit")
def submit_node(req: SubmitRequest):
    if req.secret_key != CONFIG["node_secret"]:
        agg_log.warning("submit rejected: invalid secret node_id=%s", req.node_id)
        raise HTTPException(status_code=401, detail="Invalid secret key")

    if req.node_id not in CONFIG["expected_nodes"]:
        agg_log.warning("submit rejected: unknown node_id=%s", req.node_id)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown node_id: {req.node_id}. "
                   f"Expected: {CONFIG['expected_nodes']}",
        )

    if req.round_num is not None and req.round_num != state["current_round"]:
        agg_log.warning(
            "submit rejected: round mismatch node_id=%s sent=%s current=%s",
            req.node_id,
            req.round_num,
            state["current_round"],
        )
        raise HTTPException(
            status_code=409,
            detail=(
                f"round_num {req.round_num} does not match aggregator "
                f"current_round {state['current_round']}"
            ),
        )

    with _state_lock:
        if req.node_id in state["submitted_nodes"]:
            return {
                "status": "already_submitted",
                "current_round": state["current_round"],
                "submitted_nodes": state["submitted_nodes"],
            }

        if state["merging"]:
            return {
                "status": "merging",
                "current_round": state["current_round"],
                "submitted_nodes": state["submitted_nodes"],
            }

        state["submitted_nodes"].append(req.node_id)
        state["last_update"] = _timestamp()
        agg_log.info(
            "submit accepted node_id=%s round=%s progress=%s/%s",
            req.node_id,
            state["current_round"],
            len(state["submitted_nodes"]),
            len(CONFIG["expected_nodes"]),
        )

        # Store node metrics
        state["node_metrics"][req.node_id] = {
            "avg_loss": req.avg_loss,
            "steps_completed": req.steps_completed,
            "round": state["current_round"],
            "submitted_at": _timestamp(),
        }

        # Log activity
        loss_suffix = (
            f" (loss: {req.avg_loss:.4f})" if req.avg_loss is not None else ""
        )
        _log_activity(
            f"{req.node_id} submitted round {state['current_round']}{loss_suffix}"
        )

        # Check if all nodes have submitted
        all_submitted = set(state["submitted_nodes"]) == set(CONFIG["expected_nodes"])

        if all_submitted:
            state["merging"] = True
            state["merge_error"] = None
            _log_activity(
                f"All nodes submitted — starting FedAvg for round {state['current_round']}"
            )
            # Capture metrics under lock before spawning thread
            round_metrics = {
                nid: state["node_metrics"].get(nid, {}).get("avg_loss")
                for nid in CONFIG["expected_nodes"]
            }
            round_steps = {
                nid: state["node_metrics"].get(nid, {}).get("steps_completed")
                for nid in CONFIG["expected_nodes"]
            }
            merge_round = state["current_round"]

            thread = threading.Thread(
                target=_background_merge,
                args=(merge_round, round_metrics, round_steps),
                daemon=True,
            )
            thread.start()

            return {
                "status": "merging",
                "current_round": state["current_round"],
                "submitted_nodes": state["submitted_nodes"],
            }

        return {
            "status": "submitted",
            "current_round": state["current_round"],
            "submitted_nodes": state["submitted_nodes"],
            "remaining": [
                n for n in CONFIG["expected_nodes"]
                if n not in state["submitted_nodes"]
            ],
        }


@app.post("/reset")
def reset_state(req: ResetRequest):
    if req.secret_key != _effective_reset_secret():
        agg_log.warning("reset rejected: invalid secret")
        raise HTTPException(status_code=401, detail="Invalid secret key")

    agg_log.info("state_reset")
    state["current_round"] = 1
    state["submitted_nodes"] = []
    state["history"] = []
    state["last_update"] = _timestamp()
    state["node_metrics"] = {}
    state["activity_log"] = []
    state["merging"] = False
    state["merge_error"] = None

    _log_activity("State reset to round 1")
    _persist_state_to_hub()

    return {"status": "reset", "current_round": 1}


# ---------------------------------------------------------------------------
# Gradio dashboard
# ---------------------------------------------------------------------------

def _round_progress_md():
    """Round progress bar and summary."""
    total = len(CONFIG["expected_nodes"])
    done = len(state["submitted_nodes"])
    pct = int((done / total) * 100) if total else 0
    bar = "█" * done + "░" * (total - done)

    return (
        f"## Round {state['current_round']}\n\n"
        f"**Progress:** {bar}  {done}/{total} nodes ({pct}%)\n\n"
        f"**Last update:** {state['last_update'] or 'Waiting for first submission'}"
    )


def _node_cards_md():
    """Visual status card for each node."""
    lines = []
    for node_id in CONFIG["expected_nodes"]:
        label = CONFIG["node_labels"].get(node_id, node_id)
        metrics = state["node_metrics"].get(node_id, {})

        if node_id in state["submitted_nodes"]:
            icon = "✅"
            status_text = "Submitted"
        else:
            icon = "⏳"
            status_text = "Waiting"

        lines.append(f"### {icon} {label}")
        lines.append(f"**Status:** {status_text}")

        if metrics.get("avg_loss") is not None:
            lines.append(f"**Avg Loss:** {metrics['avg_loss']:.4f}")
        if metrics.get("steps_completed") is not None:
            lines.append(f"**Steps:** {metrics['steps_completed']}")
        if metrics.get("submitted_at"):
            lines.append(f"**Submitted:** {metrics['submitted_at']}")

        lines.append("")

    return "\n".join(lines)


def _short_node_header(n: str) -> str:
    full = CONFIG["node_labels"].get(n, n)
    parts = full.split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return full


def _loss_history_md():
    """Per-round loss table and trend."""
    if not state["history"]:
        return "### Training Metrics\n\n_No completed rounds yet._"

    lines = ["### Training Metrics\n"]
    lines.append(
        "| Round | "
        + " | ".join(_short_node_header(n) for n in CONFIG["expected_nodes"])
        + " | Avg |"
    )
    lines.append("|---|" + "---|" * (len(CONFIG["expected_nodes"]) + 1))

    for h in state["history"]:
        node_losses = h.get("node_losses", {})
        values = []
        valid_losses = []
        for n in CONFIG["expected_nodes"]:
            loss = node_losses.get(n)
            if loss is not None:
                values.append(f"{loss:.4f}")
                valid_losses.append(loss)
            else:
                values.append("—")

        avg = f"{np.mean(valid_losses):.4f}" if valid_losses else "—"
        lines.append(f"| {h['round']} | " + " | ".join(values) + f" | {avg} |")

    # Trend indicator
    if len(state["history"]) >= 2:
        recent = state["history"][-2:]
        losses = []
        for h in recent:
            node_losses = h.get("node_losses", {})
            vals = [v for v in node_losses.values() if v is not None]
            if vals:
                losses.append(np.mean(vals))
        if len(losses) == 2:
            if losses[1] < losses[0]:
                lines.append(f"\n**Trend:** Loss decreasing ↓")
            else:
                lines.append(f"\n**Trend:** Loss increasing ↑")

    return "\n".join(lines)


def _convergence_figure():
    """Line chart: round vs per-node submitted avg loss (+ mean across nodes)."""
    if not state["history"]:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    rounds = [h["round"] for h in state["history"]]

    for nid in CONFIG["expected_nodes"]:
        ys = []
        for h in state["history"]:
            loss = h.get("node_losses", {}).get(nid)
            ys.append(loss if loss is not None else float("nan"))
        label = CONFIG["node_labels"].get(nid, nid)
        ax.plot(rounds, ys, marker="o", linewidth=1.5, label=label)

    means = []
    for h in state["history"]:
        vals = [v for v in h.get("node_losses", {}).values() if v is not None]
        means.append(float(np.mean(vals)) if vals else float("nan"))
    ax.plot(rounds, means, color="black", linestyle="--", linewidth=2, label="Mean")

    ax.set_xlabel("Round")
    ax.set_ylabel("Avg loss (submitted)")
    ax.set_title("Convergence (per node)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def _steps_bar_figure():
    """Bar chart: steps_completed per node for the last merged round."""
    if not state["history"]:
        return None

    h = state["history"][-1]
    steps_map = h.get("node_steps", {})
    labels = [_short_node_header(n) for n in CONFIG["expected_nodes"]]
    vals = [steps_map.get(n) for n in CONFIG["expected_nodes"]]
    if all(v is None for v in vals):
        return None

    fig, ax = plt.subplots(figsize=(8, 3))
    plot_vals = [0 if v is None else int(v) for v in vals]
    ax.bar(labels, plot_vals, color="steelblue")
    ax.set_ylabel("Steps completed")
    ax.set_title(f"Steps per node (merged round {h['round']})")
    fig.tight_layout()
    return fig


def _merged_adapters_md():
    """Links to merged adapter files on HF Hub."""
    repo = CONFIG["model_repo_id"]
    if not state["history"]:
        return "### Merged Adapters\n\n_No merges yet._"

    lines = ["### Merged Adapters\n"]
    for h in state["history"]:
        rnd = h["round"]
        url = f"https://huggingface.co/{repo}/blob/main/merged/round_{rnd}/adapter_model.safetensors"
        lines.append(f"- **Round {rnd}:** [{h.get('merge_result', 'View')}]({url})")

    return "\n".join(lines)


def _activity_log_md():
    """Recent activity feed."""
    if not state["activity_log"]:
        return "### Activity Log\n\n_No activity yet._"

    lines = ["### Activity Log\n"]
    for entry in reversed(state["activity_log"][-15:]):
        t = entry["time"].split("T")[1].split(".")[0] if "T" in entry["time"] else entry["time"]
        lines.append(f"- `{t}` {entry['message']}")

    return "\n".join(lines)


# --- Build the Gradio UI ---

with gr.Blocks(title="PEFT Aggregator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Distributed PEFT Fine-Tuning — Aggregator Dashboard")

    with gr.Row():
        progress_display = gr.Markdown(_round_progress_md)

    with gr.Row():
        with gr.Column(scale=2):
            node_cards = gr.Markdown(_node_cards_md)
        with gr.Column(scale=3):
            loss_display = gr.Markdown(_loss_history_md)

    with gr.Row():
        convergence_plot = gr.Plot(label="Convergence (avg loss per round)")
        steps_plot = gr.Plot(label="Steps completed (last merged round)")

    with gr.Row():
        with gr.Column():
            adapters_display = gr.Markdown(_merged_adapters_md)
        with gr.Column():
            activity_display = gr.Markdown(_activity_log_md)

    refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="primary")
    _dashboard_refresh = lambda: (
        _round_progress_md(),
        _node_cards_md(),
        _loss_history_md(),
        _convergence_figure(),
        _steps_bar_figure(),
        _merged_adapters_md(),
        _activity_log_md(),
    )
    _dashboard_outputs = [
        progress_display,
        node_cards,
        loss_display,
        convergence_plot,
        steps_plot,
        adapters_display,
        activity_display,
    ]

    refresh_btn.click(fn=_dashboard_refresh, outputs=_dashboard_outputs)
    demo.load(fn=_dashboard_refresh, outputs=_dashboard_outputs)

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
