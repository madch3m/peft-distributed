"""
app.py — Aggregator Space for Distributed PEFT Fine-Tuning.

Gradio dashboard + FastAPI endpoints:
    /submit  — Node reports round completion
    /status  — Current aggregation state (JSON)
    /reset   — Reset to round 1

FedAvg pipeline: when all 3 nodes submit, averages adapter states on HF Hub.
"""

import os
import json
import datetime

import gradio as gr
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import HfApi, hf_hub_download, upload_file
from safetensors.torch import load_file, save_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG = {
    "expected_nodes": ["node_a", "node_b", "node_c"],
    "node_secret": os.environ.get("NODE_SECRET", "local_test_secret"),
    "hf_token": os.environ.get("HF_TOKEN", ""),
    "model_repo_id": os.environ.get("MODEL_REPO_ID", ""),
    "node_labels": {
        "node_a": "Node A (Layers 0–10)",
        "node_b": "Node B (Layers 11–21)",
        "node_c": "Node C (Layers 22–31)",
    },
}

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
}


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

def fedavg_merge() -> str:
    """Download adapter states from all nodes, average, upload merged checkpoint."""
    api = HfApi(token=CONFIG["hf_token"])
    repo = CONFIG["model_repo_id"]
    rnd = state["current_round"]

    if not repo or not CONFIG["hf_token"]:
        return "Skipped merge — MODEL_REPO_ID or HF_TOKEN not set."

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
            return f"FedAvg failed — could not load {node_id}: {e}"

    # Average all tensors
    merged = {}
    keys = adapter_states[0].keys()
    for key in keys:
        stacked = [s[key].float() for s in adapter_states]
        merged[key] = sum(stacked) / len(stacked)

    # Save merged checkpoint
    merged_path = "/tmp/merged_adapter_model.safetensors"
    save_file(merged, merged_path)

    api.upload_file(
        path_or_fileobj=merged_path,
        path_in_repo=f"merged/round_{rnd}/adapter_model.safetensors",
        repo_id=repo,
        token=CONFIG["hf_token"],
        commit_message=f"FedAvg merge — round {rnd}",
    )

    # Update training_state.json
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

    return f"FedAvg complete for round {rnd}. Advanced to round {rnd + 1}."


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------

app = FastAPI(title="PEFT Aggregator")


class SubmitRequest(BaseModel):
    node_id: str
    secret_key: str
    round_num: int | None = None
    avg_loss: float | None = None
    steps_completed: int | None = None


class ResetRequest(BaseModel):
    secret_key: str


@app.get("/status")
def get_status():
    return {
        "current_round": state["current_round"],
        "submitted_nodes": state["submitted_nodes"],
        "expected_nodes": CONFIG["expected_nodes"],
        "remaining": [
            n for n in CONFIG["expected_nodes"]
            if n not in state["submitted_nodes"]
        ],
        "last_update": state["last_update"],
    }


@app.post("/submit")
def submit_node(req: SubmitRequest):
    if req.secret_key != CONFIG["node_secret"]:
        raise HTTPException(status_code=401, detail="Invalid secret key")

    if req.node_id not in CONFIG["expected_nodes"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown node_id: {req.node_id}. "
                   f"Expected: {CONFIG['expected_nodes']}",
        )

    if req.node_id in state["submitted_nodes"]:
        return {
            "status": "already_submitted",
            "current_round": state["current_round"],
            "submitted_nodes": state["submitted_nodes"],
        }

    state["submitted_nodes"].append(req.node_id)
    state["last_update"] = _timestamp()

    # Store node metrics
    state["node_metrics"][req.node_id] = {
        "avg_loss": req.avg_loss,
        "steps_completed": req.steps_completed,
        "round": state["current_round"],
        "submitted_at": _timestamp(),
    }

    # Log activity
    _log_activity(f"{req.node_id} submitted round {state['current_round']}"
                  + (f" (loss: {req.avg_loss:.4f})" if req.avg_loss else ""))

    # Check if all nodes have submitted
    if set(state["submitted_nodes"]) == set(CONFIG["expected_nodes"]):
        _log_activity(f"All nodes submitted — starting FedAvg for round {state['current_round']}")
        merge_result = fedavg_merge()

        # Capture per-node losses in history
        round_metrics = {
            nid: state["node_metrics"].get(nid, {}).get("avg_loss")
            for nid in CONFIG["expected_nodes"]
        }

        state["history"].append({
            "round": state["current_round"],
            "completed_at": _timestamp(),
            "merge_result": merge_result,
            "node_losses": round_metrics,
        })

        _log_activity(f"Round {state['current_round']} FedAvg complete")

        state["current_round"] += 1
        state["submitted_nodes"] = []

        return {
            "status": "round_complete",
            "merge_result": merge_result,
            "new_round": state["current_round"],
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
    if req.secret_key != CONFIG["node_secret"]:
        raise HTTPException(status_code=401, detail="Invalid secret key")

    state["current_round"] = 1
    state["submitted_nodes"] = []
    state["history"] = []
    state["last_update"] = _timestamp()
    state["node_metrics"] = {}
    state["activity_log"] = []

    _log_activity("State reset to round 1")

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


def _loss_history_md():
    """Per-round loss table and trend."""
    if not state["history"]:
        return "### Training Metrics\n\n_No completed rounds yet._"

    lines = ["### Training Metrics\n"]
    lines.append("| Round | " + " | ".join(
        CONFIG["node_labels"].get(n, n).split(" ")[0] + " " + CONFIG["node_labels"].get(n, n).split(" ")[1]
        for n in CONFIG["expected_nodes"]
    ) + " | Avg |")
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
        with gr.Column():
            adapters_display = gr.Markdown(_merged_adapters_md)
        with gr.Column():
            activity_display = gr.Markdown(_activity_log_md)

    refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="primary")
    refresh_btn.click(
        fn=lambda: (
            _round_progress_md(),
            _node_cards_md(),
            _loss_history_md(),
            _merged_adapters_md(),
            _activity_log_md(),
        ),
        outputs=[progress_display, node_cards, loss_display, adapters_display, activity_display],
    )

app = gr.mount_gradio_app(app, demo, path="/")
