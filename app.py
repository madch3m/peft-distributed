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
}

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

state = {
    "current_round": 1,
    "submitted_nodes": [],
    "history": [],
    "last_update": None,
}


def _timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


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

    # Check if all nodes have submitted
    if set(state["submitted_nodes"]) == set(CONFIG["expected_nodes"]):
        merge_result = fedavg_merge()

        state["history"].append({
            "round": state["current_round"],
            "completed_at": _timestamp(),
            "merge_result": merge_result,
        })

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

    return {"status": "reset", "current_round": 1}


# ---------------------------------------------------------------------------
# Gradio dashboard
# ---------------------------------------------------------------------------

def dashboard_status():
    remaining = [
        n for n in CONFIG["expected_nodes"]
        if n not in state["submitted_nodes"]
    ]
    lines = [
        f"## Round {state['current_round']}",
        "",
        f"**Submitted:** {', '.join(state['submitted_nodes']) or 'none'}",
        f"**Remaining:** {', '.join(remaining) or 'none'}",
        f"**Last update:** {state['last_update'] or 'N/A'}",
        "",
        "### History",
    ]
    if state["history"]:
        for h in state["history"][-5:]:
            lines.append(
                f"- Round {h['round']} completed at {h['completed_at']}"
            )
    else:
        lines.append("_No completed rounds yet._")

    return "\n".join(lines)


with gr.Blocks(title="PEFT Aggregator") as demo:
    gr.Markdown("# Distributed PEFT Fine-Tuning — Aggregator Dashboard")
    status_display = gr.Markdown(dashboard_status)
    refresh_btn = gr.Button("Refresh")
    refresh_btn.click(fn=dashboard_status, outputs=status_display)

app = gr.mount_gradio_app(app, demo, path="/")
