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
| `/health` | GET | Liveness probe (`{"ok": true}`); no Hub I/O |
| `/submit` | POST | Node submits round completion |
| `/status` | GET | Current aggregation state (JSON) |
| `/reset` | POST | Reset to round 1 (see `ADMIN_SECRET` below) |

## Secrets Required

| Secret | Description |
|---|---|
| `HF_TOKEN` | HuggingFace write-access token |
| `MODEL_REPO_ID` | Target model repo (e.g. `your-org/your-model-repo`) |
| `NODE_SECRET` | Shared secret — must match all Colab nodes |
| `ADMIN_SECRET` | *(Optional)* If set, `POST /reset` requires this value instead of `NODE_SECRET` |
| `STATUS_READ_SECRET` | *(Optional)* If set, `GET /status` requires header `X-Status-Secret: <value>` |

### Where to set each secret

#### 1. Hugging Face Space (Aggregator)

All three secrets must be added to the Space. Go to your Space's **Settings > Repository secrets** and add:

- `HF_TOKEN` — Your HuggingFace token with **write** access
- `MODEL_REPO_ID` — The HF repo where adapter weights are stored (e.g. `your-username/your-model-repo`)
- `NODE_SECRET` — Any passphrase you choose; every Colab node must use this same value
- `ADMIN_SECRET` — *(Optional)* Separate passphrase for **`POST /reset` only**. If unset, reset uses `NODE_SECRET`.
- `STATUS_READ_SECRET` — *(Optional)* If set, clients must send **`X-Status-Secret`** on **`GET /status`**. The Gradio dashboard still works (it reads in-process state, not HTTP `/status`).

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

If you pass **`round_num`**, it must equal the aggregator’s **`current_round`** (see `GET /status`); otherwise the Space returns **409**.

When the last node completes a round, the client may receive **`merge_failed`** if FedAvg or Hub upload fails. **`aggregator_client.notify_aggregator`** then raises **`AggregatorMergeFailed`** with **`e.payload["merge_result"]`** explaining the error (fix Hub paths or permissions, then have nodes submit again).

Nodes do **not** need `MODEL_REPO_ID` — only the aggregator uses it to download/upload adapter weights.

#### 3. Local testing

Export the variables before running the server:

```bash
export HF_TOKEN="hf_..."
export MODEL_REPO_ID="your-username/your-model-repo"
export NODE_SECRET="local_test_secret"
```

Without these, the app defaults to empty strings (merge is skipped) and `"local_test_secret"` for the node secret.

## Operator notes

- **Secrets:** Never commit `HF_TOKEN`, `NODE_SECRET`, or tokens in git remotes. Use Space **Repository secrets** and a local env or credential helper.
- **Reset:** `POST /reset` with JSON `{"secret_key": "<ADMIN_SECRET or NODE_SECRET>"}` clears round state to 1. If `ADMIN_SECRET` is set on the Space, use that; otherwise use `NODE_SECRET`.
- **Protected status:** When `STATUS_READ_SECRET` is set, pass the same value as header `X-Status-Secret` (see `aggregator_client.check_aggregator` / `poll_for_next_round` argument `status_secret`, and notebook `CONFIG["status_read_secret"]`).
- **Rate limits:** `POST /submit` is limited per client IP (first `X-Forwarded-For` hop when present). Override with **`RATE_LIMIT_SUBMIT_MAX`** (default 120) and **`RATE_LIMIT_SUBMIT_WINDOW_SEC`** (default 60).
- **Logs:** Set **`LOG_LEVEL`** (e.g. `DEBUG`, `INFO`, `WARNING`) for the `peft.aggregator` logger on stdout.
- **Public Space:** `GET /status` is world-readable on a public Space; use a private Space if round visibility matters.

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
