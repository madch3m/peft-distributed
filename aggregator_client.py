"""
aggregator_client.py — Node-side helper for communicating with the aggregator Space.

Functions:
    notify_aggregator()   — Tell the aggregator this node finished its round
    poll_for_next_round() — Block until FedAvg is done and next round begins
    check_aggregator()    — Quick health / status check
"""

import time
import requests


def notify_aggregator(
    aggregator_url: str,
    node_id: str,
    secret_key: str,
    round_num: int | None = None,
    timeout: int = 30,
) -> dict:
    """Notify the aggregator that this node has finished its current round.

    Args:
        aggregator_url: Base URL of the aggregator Space.
        node_id: One of 'node_a', 'node_b', 'node_c'.
        secret_key: Shared secret for authentication.
        round_num: Optional current round number for verification.
        timeout: Request timeout in seconds.

    Returns:
        JSON response from the aggregator.
    """
    payload = {
        "node_id": node_id,
        "secret_key": secret_key,
    }
    if round_num is not None:
        payload["round_num"] = round_num

    url = f"{aggregator_url.rstrip('/')}/submit"
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def poll_for_next_round(
    aggregator_url: str,
    current_round: int,
    poll_interval: int = 30,
    max_wait: int = 1800,
) -> dict:
    """Block until the aggregator advances past the current round.

    Args:
        aggregator_url: Base URL of the aggregator Space.
        current_round: The round we just finished.
        poll_interval: Seconds between status checks.
        max_wait: Maximum seconds to wait before raising TimeoutError.

    Returns:
        Status dict from the aggregator once it advances.

    Raises:
        TimeoutError: If max_wait is exceeded.
    """
    url = f"{aggregator_url.rstrip('/')}/status"
    elapsed = 0

    while elapsed < max_wait:
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            status = resp.json()
            agg_round = status.get("current_round", 0)
            if agg_round > current_round:
                print(f"[poll] Aggregator advanced to round {agg_round}")
                return status
        except requests.RequestException as e:
            print(f"[poll] Request error: {e}")

        print(
            f"[poll] Waiting for round {current_round + 1} "
            f"({elapsed}/{max_wait}s elapsed)..."
        )
        time.sleep(poll_interval)
        elapsed += poll_interval

    raise TimeoutError(
        f"Aggregator did not advance past round {current_round} "
        f"within {max_wait}s. Check if other nodes have finished."
    )


def check_aggregator(aggregator_url: str, timeout: int = 10) -> dict:
    """Quick health check — returns the aggregator status JSON.

    Args:
        aggregator_url: Base URL of the aggregator Space.
        timeout: Request timeout in seconds.

    Returns:
        Status dict with current_round, submitted_nodes, etc.
    """
    url = f"{aggregator_url.rstrip('/')}/status"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()
