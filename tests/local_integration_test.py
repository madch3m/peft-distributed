"""
Local integration test — simulates 3 notebook nodes hitting the aggregator.

Usage:
    1. Start the aggregator:   python app.py
    2. Run this script:        python tests/local_integration_test.py

No HF credentials needed — FedAvg merge is skipped (MODEL_REPO_ID empty).
"""

import sys
import time
import threading

sys.path.insert(0, ".")

from aggregator_client import (
    check_aggregator,
    health_aggregator,
    notify_aggregator,
    poll_for_next_round,
    reset_aggregator,
)

AGGREGATOR_URL = "http://127.0.0.1:7860"
SECRET = "local_test_secret"
NODES = ["node_a", "node_b", "node_c"]
NUM_ROUNDS = 2


def run_node(node_id, num_rounds, delay_before_submit=0):
    """Simulate a single node's training loop."""
    for rnd in range(1, num_rounds + 1):
        # Simulate training time — stagger nodes so they don't all arrive at once
        if delay_before_submit:
            time.sleep(delay_before_submit)

        print(f"[{node_id}] Round {rnd}: submitting...")
        result = notify_aggregator(
            aggregator_url=AGGREGATOR_URL,
            node_id=node_id,
            secret_key=SECRET,
            round_num=rnd,
            avg_loss=1.5 - (rnd * 0.1),  # fake decreasing loss
            steps_completed=100,
        )
        print(f"[{node_id}] Round {rnd}: submit response → {result['status']}")

        if result.get("status") != "round_complete":
            print(f"[{node_id}] Round {rnd}: polling for next round...")
            status = poll_for_next_round(
                aggregator_url=AGGREGATOR_URL,
                current_round=rnd,
                poll_interval=2,  # fast polling for local test
            )
            print(f"[{node_id}] Round {rnd}: aggregator advanced → round {status['current_round']}")

    print(f"[{node_id}] Done — all {num_rounds} rounds complete.")


def main():
    # 1. Health check
    print("=" * 60)
    print("Health check...")
    h = health_aggregator(AGGREGATOR_URL)
    print(f"  /health → {h}")

    # 2. Reset state
    print("\nResetting aggregator...")
    r = reset_aggregator(AGGREGATOR_URL, SECRET)
    print(f"  /reset → {r}")

    # 3. Check initial status
    s = check_aggregator(AGGREGATOR_URL)
    print(f"  /status → round={s['current_round']}, submitted={s['submitted_nodes']}, merging={s.get('merging')}")

    # 4. Run 3 nodes in parallel threads with staggered starts
    print("\n" + "=" * 60)
    print(f"Starting {len(NODES)} nodes for {NUM_ROUNDS} rounds...")
    print("=" * 60)

    threads = []
    for i, node_id in enumerate(NODES):
        t = threading.Thread(
            target=run_node,
            args=(node_id, NUM_ROUNDS, i * 1),  # 0s, 1s, 2s stagger
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=120)

    # 5. Final status
    print("\n" + "=" * 60)
    s = check_aggregator(AGGREGATOR_URL)
    print(f"Final status: round={s['current_round']}, submitted={s['submitted_nodes']}, merging={s.get('merging')}")
    assert s["current_round"] == NUM_ROUNDS + 1, f"Expected round {NUM_ROUNDS + 1}, got {s['current_round']}"
    assert not s.get("merging"), "Merge should be complete"
    print("ALL PASSED")


if __name__ == "__main__":
    main()
