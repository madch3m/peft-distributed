"""
aggregator_client.py — Node-side helper for communicating with the aggregator Space.

Functions:
    notify_aggregator()   — Tell the aggregator this node finished its round
    poll_for_next_round() — Block until FedAvg is done and next round begins
    check_aggregator()    — Aggregation state (GET /status)
    health_aggregator()   — Liveness (GET /health)
    reset_aggregator()    — POST /reset (use ADMIN_SECRET when Space has one)
"""

import time
from urllib.parse import urlsplit, urlunsplit

import requests


def _detail_from_response(response: requests.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict):
            d = data.get("detail")
            if d is not None:
                return str(d) if not isinstance(d, list) else str(d)
    except Exception:
        pass
    text = (response.text or "").strip()
    if text:
        return text[:800]
    return response.reason or ""


def _raise_for_aggregator_response(response: requests.Response, *, what: str) -> None:
    """Raise HTTPError with server ``detail`` and hints for common operator mistakes."""
    if response.status_code < 400:
        return
    detail = _detail_from_response(response)
    msg = f"{what}: HTTP {response.status_code}"
    if detail:
        msg += f" — {detail}"
    if response.status_code == 401:
        msg += (
            ". Hint: POST /submit needs JSON ``secret_key`` equal to the Space "
            "``NODE_SECRET``. GET /status needs header ``X-Status-Secret`` when the "
            "Space sets ``STATUS_READ_SECRET`` — pass ``status_secret=...`` from "
            "``poll_for_next_round`` / ``check_aggregator``, or set "
            "``CONFIG['status_read_secret']`` in the notebook to match the Space."
        )
    raise requests.HTTPError(msg, response=response)


def _normalize_aggregator_base_url(url: str) -> str:
    """Strip whitespace/slashes and ensure an HTTP(S) scheme for requests.

    Colab configs often omit ``https://``, which causes requests to raise
    ``MissingSchema``.

    Also fixes: ``https:host`` (missing ``//`` after the scheme); pasting
    ``https://OWNER/SPACE.hf.space`` (slash) instead of ``https://OWNER-SPACE.hf.space``
    (otherwise the client treats ``OWNER`` as the hostname and DNS fails).
    """
    base = url.strip().rstrip("/")
    if not base:
        raise ValueError("aggregator_url is empty")

    low = base.lower()
    # ``https:host`` / ``http:host`` (missing ``//``) does not contain ``://``;
    # blindly prefixing ``https://`` would produce ``https://https:host`` and
    # break urllib3 (InvalidURL).
    if low.startswith("https:") and not low.startswith("https://"):
        base = "https://" + base[6:].lstrip("/")
    elif low.startswith("http:") and not low.startswith("http://"):
        base = "http://" + base[5:].lstrip("/")
    elif "://" not in base:
        base = "https://" + base.lstrip("/")

    parts = urlsplit(base)
    scheme, netloc, path, query, fragment = parts
    path_clean = path.rstrip("/")

    # Wrong: https://dev-the-dev91/instruction-fine-tuning-budget.hf.space
    # Right: https://dev-the-dev91-instruction-fine-tuning-budget.hf.space
    if path_clean.startswith("/"):
        leaf = path_clean.lstrip("/")
        if leaf.endswith(".hf.space") and not netloc.endswith(".hf.space"):
            netloc = f"{netloc}-{leaf}"
            path = ""

    base = urlunsplit((scheme, netloc, path, query, fragment)).rstrip("/")
    return base


def _status_headers(status_secret: str | None) -> dict[str, str]:
    if status_secret:
        return {"X-Status-Secret": status_secret}
    return {}


class AggregatorMergeFailed(RuntimeError):
    """Raised when the Space returns merge_failed (FedAvg or Hub step failed)."""

    def __init__(self, payload: dict):
        self.payload = payload
        msg = payload.get("merge_result") or "merge_failed"
        super().__init__(msg)


def notify_aggregator(
    aggregator_url: str,
    node_id: str,
    secret_key: str,
    round_num: int | None = None,
    avg_loss: float | None = None,
    eval_loss: float | None = None,
    perplexity: float | None = None,
    steps_completed: int | None = None,
    timeout: int = 30,
) -> dict:
    """Notify the aggregator that this node has finished its current round.

    Args:
        aggregator_url: Base URL of the aggregator Space (with or without
            ``https://``; missing scheme defaults to ``https://``).
        node_id: One of 'node_a', 'node_b', 'node_c'.
        secret_key: Shared secret for authentication.
        round_num: Optional current round number for verification.
        avg_loss: Optional mean training loss for this round (shown on Space dashboard).
        eval_loss: Optional evaluation loss on held-out data for this round.
        perplexity: Optional perplexity metric for this round.
        steps_completed: Optional step count for this round (e.g. CONFIG steps_per_round).
        timeout: Request timeout in seconds.

    Returns:
        JSON response from the aggregator.

    Raises:
        AggregatorMergeFailed: If status is merge_failed (fix Hub files and resubmit).
    """
    payload = {
        "node_id": node_id,
        "secret_key": secret_key,
    }
    if round_num is not None:
        payload["round_num"] = round_num
    if avg_loss is not None:
        payload["avg_loss"] = avg_loss
    if steps_completed is not None:
        payload["steps_completed"] = steps_completed
    if eval_loss is not None:
        payload["eval_loss"] = eval_loss
    if perplexity is not None:
        payload["perplexity"] = perplexity

    base = _normalize_aggregator_base_url(aggregator_url)
    url = f"{base}/submit"
    response = requests.post(url, json=payload, timeout=timeout)
    if response.status_code == 404:
        raise RuntimeError(
            f"No route at {url!r} (404). Use the Space app host that serves the API "
            "(typically https://YOUR_SPACE_NAME.hf.space), not the huggingface.co/spaces "
            "HTML page URL. Path must be exactly /submit."
        )
    _raise_for_aggregator_response(response, what="POST /submit")
    data = response.json()
    if data.get("status") == "merge_failed":
        raise AggregatorMergeFailed(data)
    return data


def poll_for_next_round(
    aggregator_url: str,
    current_round: int,
    poll_interval: int = 10,
    max_wait: int = 1800,
    status_secret: str | None = None,
) -> dict:
    """Block until the aggregator advances past the current round.

    Args:
        aggregator_url: Base URL of the aggregator Space (with or without
            ``https://``; missing scheme defaults to ``https://``).
        current_round: The round we just finished.
        poll_interval: Seconds between status checks (default 10).
        max_wait: Maximum seconds to wait before raising TimeoutError.
        status_secret: If the Space sets STATUS_READ_SECRET, pass it here
            (sent as X-Status-Secret on GET /status).

    Returns:
        Status dict from the aggregator once it advances.

    Raises:
        TimeoutError: If max_wait is exceeded.
        AggregatorMergeFailed: If the aggregator reports a merge error.
    """
    base = _normalize_aggregator_base_url(aggregator_url)
    url = f"{base}/status"
    headers = _status_headers(status_secret)
    elapsed = 0

    while elapsed < max_wait:
        try:
            resp = requests.get(url, timeout=15, headers=headers)
            _raise_for_aggregator_response(resp, what="GET /status")
            status = resp.json()

            # Check for merge failure
            merge_error = status.get("merge_error")
            if merge_error:
                raise AggregatorMergeFailed({"merge_result": merge_error})

            agg_round = status.get("current_round", 0)
            if agg_round > current_round:
                print(f"[poll] Aggregator advanced to round {agg_round}")
                return status

            if status.get("merging"):
                print(
                    f"[poll] FedAvg merge in progress "
                    f"({elapsed}/{max_wait}s elapsed)..."
                )
            else:
                print(
                    f"[poll] Waiting for round {current_round + 1} "
                    f"({elapsed}/{max_wait}s elapsed)..."
                )
        except requests.RequestException as e:
            print(f"[poll] Request error: {e}")

        time.sleep(poll_interval)
        elapsed += poll_interval

    raise TimeoutError(
        f"Aggregator did not advance past round {current_round} "
        f"within {max_wait}s. Check if other nodes have finished."
    )


def check_aggregator(
    aggregator_url: str,
    timeout: int = 10,
    *,
    status_secret: str | None = None,
) -> dict:
    """Return aggregation state from GET /status (round, submitted_nodes, …)."""
    url = f"{_normalize_aggregator_base_url(aggregator_url)}/status"
    response = requests.get(
        url,
        timeout=timeout,
        headers=_status_headers(status_secret),
    )
    _raise_for_aggregator_response(response, what="GET /status")
    return response.json()


def reset_aggregator(
    aggregator_url: str,
    secret_key: str,
    timeout: int = 30,
) -> dict:
    """Reset aggregator to round 1. Use ADMIN_SECRET if the Space defines it, else NODE_SECRET."""
    url = f"{_normalize_aggregator_base_url(aggregator_url)}/reset"
    response = requests.post(
        url,
        json={"secret_key": secret_key},
        timeout=timeout,
    )
    _raise_for_aggregator_response(response, what="POST /reset")
    return response.json()


def health_aggregator(aggregator_url: str, timeout: int = 10) -> dict:
    """Liveness probe via GET /health; does not depend on training state."""
    url = f"{_normalize_aggregator_base_url(aggregator_url)}/health"
    response = requests.get(url, timeout=timeout)
    _raise_for_aggregator_response(response, what="GET /health")
    return response.json()
