---
title: PEFT Distributed Aggregator
emoji: "🔗"
colorFrom: blue
colorTo: green
sdk: docker
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

### Where to set each secret

#### 1. Hugging Face Space (Aggregator)

All three secrets must be added to the Space. Go to your Space's **Settings > Repository secrets** and add:

- `HF_TOKEN` — Your HuggingFace token with **write** access
- `MODEL_REPO_ID` — The HF repo where adapter weights are stored (e.g. `your-username/your-model-repo`)
- `NODE_SECRET` — Any passphrase you choose; every Colab node must use this same value

These are read as environment variables in `app.py` (`os.environ.get(...)`).

#### 2. Colab Nodes (Clients)

Each Colab notebook only needs to know the `NODE_SECRET` (must match the Space) and the aggregator URL. Set them in a cell:

```python
AGGREGATOR_URL = "https://your-username-your-space.hf.space"
NODE_SECRET = "my-super-secret-123"  # must match the Space secret
```

Then call the aggregator client:

```python
from aggregator_client import notify_aggregator

notify_aggregator(AGGREGATOR_URL, "node_a", NODE_SECRET, round_num=1)
```

Nodes do **not** need `MODEL_REPO_ID` — only the aggregator uses it to download/upload adapter weights.

#### 3. Local testing

Export the variables before running the server:

```bash
export HF_TOKEN="hf_..."
export MODEL_REPO_ID="your-username/your-model-repo"
export NODE_SECRET="local_test_secret"
```

Without these, the app defaults to empty strings (merge is skipped) and `"local_test_secret"` for the node secret.

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
