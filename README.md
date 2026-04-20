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

Use the Space’s **direct app host** as `AGGREGATOR_URL`: **`https://<owner>-<space-name>.hf.space`** (open your Space → use the **App** tab URL, or copy from the Space card). It is **not** `huggingface.co/spaces/user/repo`, **not** `https://<owner>/<space>.hf.space` (a slash makes the client try to resolve `<owner>` as the hostname and DNS fails), and **not** a model repo id. The client accepts the host **with or without** `https://` (missing scheme defaults to `https://`), fixes **`https:host`** typos (missing slashes after the scheme), and **rewrites** the common `owner/space.hf.space` mistake into `owner-space.hf.space`. API paths are `/submit`, `/status`, `/health`, `/reset`.

Then call the aggregator client:

```python
from aggregator_client import notify_aggregator

notify_aggregator(
    AGGREGATOR_URL,
    "node_a",
    NODE_SECRET,
    round_num=1,
    avg_loss=0.42,           # optional: mean training loss for this round (dashboard table + charts)
    steps_completed=100,     # optional: steps trained this round (e.g. CONFIG["steps_per_round"])
    eval_loss = 0.51        # optional: held-out evaluation loss
    perplexity = 98.4,      # optional: perplexity (exp of eval_loss)
)
```

If you pass **`round_num`**, it must equal the aggregator’s **`current_round`** (see `GET /status`); otherwise the Space returns **409**.

For **convergence plots** on the Space (`GET /`), each node should submit **`avg_loss`** (round-average loss) and **`steps_completed`** **every round** after training that round. The Colab notebook calls `round_end_sync(..., avg_loss=..., steps_completed=...)` at the end of each round for this purpose.

When the last node completes a round, the client may receive **`merge_failed`** if FedAvg or Hub upload fails. **`aggregator_client.notify_aggregator`** then raises **`AggregatorMergeFailed`** with **`e.payload["merge_result"]`** explaining the error (fix Hub paths or permissions, then have nodes submit again).

Nodes do **not** need `MODEL_REPO_ID` — only the aggregator uses it to download/upload adapter weights.

#### 3. Local or Space runtime env

For **local** runs, export `HF_TOKEN`, `MODEL_REPO_ID`, and `NODE_SECRET` (or rely on defaults described in **Testing** below). On the **Hugging Face Space**, set the same keys as **Repository secrets**.

## Operator notes

- **Dashboard:** The Gradio UI shows per-node cards, a per-round loss table, **matplotlib** convergence and step-count plots, merged-adapter links, and an activity log. Use **Refresh** to update; metrics require nodes to POST **`avg_loss`** / **`steps_completed`** / **`eval_loss`** / **`perplexity`** on **`/submit`** (see `notify_aggregator` keyword arguments above).
- **Secrets:** Never commit `HF_TOKEN`, `NODE_SECRET`, or tokens in git remotes. Use Space **Repository secrets** and a local env or credential helper.
- **Reset:** `POST /reset` with JSON `{"secret_key": "<ADMIN_SECRET or NODE_SECRET>"}` clears round state to 1. If `ADMIN_SECRET` is set on the Space, use that; otherwise use `NODE_SECRET`.
- **Protected status:** When `STATUS_READ_SECRET` is set, pass the same value as header `X-Status-Secret` (see `aggregator_client.check_aggregator` / `poll_for_next_round` argument `status_secret`, and notebook `CONFIG["status_read_secret"]`).
- **401 Unauthorized:** (1) **`POST /submit`** — JSON `secret_key` must match the Space **`NODE_SECRET`** (same string in every Colab `CONFIG["node_secret"]`). (2) **`GET /status`** — if the Space defines **`STATUS_READ_SECRET`**, clients must send that value (notebook `CONFIG["status_read_secret"]`, or clear `STATUS_READ_SECRET` on the Space if you do not need it). **`GET /health`** has no secret. A **private** Hugging Face Space can also return 401 at the edge before your app runs — open the Space in the browser while logged in, or check Space visibility settings.
- **Rate limits:** `POST /submit` is limited per client IP (first `X-Forwarded-For` hop when present). Override with **`RATE_LIMIT_SUBMIT_MAX`** (default 120) and **`RATE_LIMIT_SUBMIT_WINDOW_SEC`** (default 60).
- **Logs:** Set **`LOG_LEVEL`** (e.g. `DEBUG`, `INFO`, `WARNING`) for the `peft.aggregator` logger on stdout.
- **Public Space:** `GET /status` is world-readable on a public Space; use a private Space if round visibility matters.

## Architecture

- **Model:** Phi-2 (2.7B), frozen base weights
- **Method:** LoRA (r=16, alpha=32) — adapter-only training
- **Nodes:** 3 x Google Colab free T4 GPU
- **Aggregation:** FedAvg over adapter states

## Testing

### Automated tests (CI / laptop)

From the repo root, use a virtualenv with **Python 3.10+** (3.12 is fine). Install dependencies and run the suite:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pytest httpx "gradio==4.44.1"
pip install torch --index-url https://download.pytorch.org/whl/cpu
pytest tests/ -q
```

Tests exercise **FastAPI** routes (`/health`, `/status`, `/submit`, `/reset`) via `TestClient`. They set **`HF_TOKEN`** and **`MODEL_REPO_ID`** empty so **FedAvg is skipped** and no Hub network calls are required. **`aggregator_client`** URL normalization is covered in `tests/test_aggregator_client.py`.

### Run locally (same stack as the Space)

```bash
export NODE_SECRET="local_test_secret"
# Optional: real FedAvg on Hub (otherwise merge is skipped when the third node submits)
export HF_TOKEN=""
export MODEL_REPO_ID=""

pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "gradio==4.44.1"
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

Open **`http://127.0.0.1:7860/`** for the Gradio dashboard. Smoke-test JSON endpoints:

```bash
BASE="http://127.0.0.1:7860"
curl -sS "$BASE/health"
curl -sS "$BASE/status"
curl -sS -X POST "$BASE/submit" -H "Content-Type: application/json" \
  -d "{\"node_id\":\"node_a\",\"secret_key\":\"local_test_secret\"}"
```

If the Space uses **`STATUS_READ_SECRET`**, mirror that locally:

```bash
curl -sS "$BASE/status" -H "X-Status-Secret: your-secret"
```

**Note:** `requirements.txt` caps **FastAPI** and **Starlette** below versions that ship **Starlette 1.x**. Gradio **4.44.x** is incompatible with that stack (the Space would return **500** on `GET /` with Jinja `unhashable type: 'dict'`). Upgrade Gradio before raising those caps.

### Docker (parity with the HF Space)

```bash
docker build -t peft-aggregator .
docker run --rm -p 7860:7860 \
  -e NODE_SECRET="local_test_secret" \
  -e HF_TOKEN="" \
  -e MODEL_REPO_ID="" \
  peft-aggregator
```

Then use the same **`curl`** examples with **`BASE=http://127.0.0.1:7860`**.

### Test the deployed Hugging Face Space

Use your Space **App** URL (hyphenated **`.hf.space`** host). Set **`BASE`** and **`SECRET`** to match **Repository secrets** on the Space.

```bash
BASE="https://YOUR_OWNER-YOUR_SPACE_NAME.hf.space"
SECRET="your-node-secret-from-space-settings"

curl -sS "$BASE/health"
```

**`GET /status`** — if **`STATUS_READ_SECRET`** is set on the Space, add the header; otherwise a public Space returns JSON without auth:

```bash
curl -sS "$BASE/status"
# or: curl -sS "$BASE/status" -H "X-Status-Secret: $STATUS_READ_SECRET"
```

**`POST /submit`** — any node can submit independently; the first responses are **`"status":"submitted"`** with **`remaining`** until all three IDs have submitted for the current round. Omit **`round_num`** for a quick smoke test, or set **`round_num`** to the value returned by **`/status`** as **`current_round`** (mismatch → **409**).

```bash
for id in node_a node_b node_c; do
  curl -sS -X POST "$BASE/submit" -H "Content-Type: application/json" \
    -d "{\"node_id\":\"$id\",\"secret_key\":\"$SECRET\",\"avg_loss\":1.0,\"steps_completed\":10}"
  echo
done
```

After the third submit you should see **`"status":"round_complete"`** (with **`HF_TOKEN`/`MODEL_REPO_ID`** configured and valid adapter files on the Hub) or a message that merge was **skipped** / **`merge_failed`** if Hub setup is incomplete — both outcomes confirm the Space is running the aggregation logic.

**`POST /reset`** — use **`ADMIN_SECRET`** if the Space defines it, else **`NODE_SECRET`**:

```bash
curl -sS -X POST "$BASE/reset" -H "Content-Type: application/json" \
  -d "{\"secret_key\":\"$SECRET\"}"
```

**Python** — the same checks with **`aggregator_client`**:

```python
from aggregator_client import check_aggregator, health_aggregator, notify_aggregator

BASE = "https://YOUR_OWNER-YOUR_SPACE_NAME.hf.space"
SECRET = "your-node-secret"

health_aggregator(BASE)
check_aggregator(BASE, status_secret=None)  # or status_secret="..." if configured
notify_aggregator(BASE, "node_a", SECRET, round_num=1)
```

**Private Space:** you may need to be logged into Hugging Face in a browser to open **`/`**; API calls from Colab or scripts still use **`NODE_SECRET`** on **`/submit`** and **`X-Status-Secret`** on **`/status`** when applicable — they do not use your HF login cookie.

## Loading the Fine-Tuned Model

After training completes, the merged LoRA adapter for each round is stored on Hub at `merged/round_N/adapter_model.safetensors`. To load the model for inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file
import torch
import json

REPO_ID = "Dev-the-dev91/instruction-fine-tuning-budget"
HF_TOKEN = "hf_YOUR_TOKEN"
BASE_MODEL = "microsoft/phi-2"

# Find the latest merged round automatically
api = HfApi(token=HF_TOKEN)
state_path = hf_hub_download(repo_id=REPO_ID, filename="aggregator_state.json", token=HF_TOKEN)
with open(state_path) as f:
    state = json.load(f)
latest_round = state["current_round"] - 1
print(f"Latest merged round: {latest_round}")

# Load base model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa",
)

# Attach LoRA with same config used during training
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "dense", "fc1", "fc2"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Download and apply the merged adapter
merged_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=f"merged/round_{latest_round}/adapter_model.safetensors",
    token=HF_TOKEN,
)
merged_state = load_file(merged_path)
model_params = dict(model.named_parameters())
loaded = 0
for key, tensor in merged_state.items():
    if key in model_params:
        model_params[key].data.copy_(tensor.to(model_params[key].device))
        loaded += 1
print(f"Loaded {loaded} adapter params from round {latest_round}")

# Generate
model.eval()
prompt = "### Instruction:\nExplain photosynthesis in three sentences.\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

You can also load a specific round by setting `latest_round = N` instead of reading from `aggregator_state.json`.

Deadline: April 20, 2026 · Lead target April 13
