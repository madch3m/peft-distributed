---
title: PEFT Distributed Aggregator
emoji: "🔗"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
---

# Distributed PEFT Fine-Tuning — Aggregator Space

3-node distributed LoRA fine-tuning of Phi-2 with FedAvg aggregation.

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Gradio dashboard |
| `/submit` | POST | Node submits round completion |
| `/status` | GET | Current aggregation state (JSON) |
| `/reset` | POST | Reset to round 1 |

## Secrets Required

| Secret | Description |
|---|---|
| `HF_TOKEN` | HuggingFace write-access token |
| `MODEL_REPO_ID` | Target model repo (e.g. `your-org/your-model-repo`) |
| `NODE_SECRET` | Shared secret — must match all Colab nodes |

## Architecture

- **Model:** Phi-2 (2.7B), frozen base weights
- **Method:** LoRA (r=16, alpha=32) — adapter-only training
- **Nodes:** 3 x Google Colab free T4 GPU
- **Aggregation:** FedAvg over adapter states

## Local Testing

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

Deadline: April 20, 2026 · Lead target April 13
