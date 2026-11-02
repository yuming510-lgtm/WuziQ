from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import src.research.alphazero_g.v3_budget as budget


def _record_cpu_event_after_cache_load(root: str, barrier, queue) -> None:
    import src.research.alphazero_g.v3_budget as worker_budget
    worker_budget._load_cpu_cache(Path(root))
    barrier.wait(timeout=10)
    queue.put(worker_budget.record_v3_cpu_event(root, "evaluation:shared", 1.25,
                                                 category="evaluation"))


def test_cpu_event_ledger_is_durable_and_idempotent(tmp_path):
    assert budget.record_v3_cpu_event(tmp_path, "selfplay:g-1", 2.5, category="selfplay")
    assert not budget.record_v3_cpu_event(tmp_path, "selfplay:g-1", 2.5, category="selfplay")
    assert budget.cpu_seconds_committed(tmp_path) == 2.5
    state = json.loads((tmp_path / "research/g_v3/results/runtime/cpu_budget.json").read_text())
    assert state["event_count"] == 1
    assert state["total_cpu_seconds"] == 2.5


def test_cpu_event_ledger_reconciles_unrecorded_game_after_restart(tmp_path):
    source = tmp_path / "research/g_v3/results/selfplay/G0-seed-510901.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"game_id": "g-2", "runner_cpu_seconds": 4.25}) + "\n")

    budget._CPU_CACHES.pop(str(tmp_path.resolve()), None)
    budget._RECONCILED_ROOTS.discard(str(tmp_path.resolve()))
    assert budget.cpu_seconds_committed(tmp_path) == 4.25
    assert budget.cpu_seconds_committed(tmp_path) == 4.25
    rows = (tmp_path / "research/g_v3/results/runtime/cpu_events.jsonl").read_text().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0]) == {
        "category": "selfplay", "cpu_seconds": 4.25, "event_id": "selfplay:g-2",
    }


def test_incomplete_cpu_event_tail_is_discarded_before_append(tmp_path):
    directory = tmp_path / "research/g_v3/results/runtime"
    directory.mkdir(parents=True)
    (directory / "cpu_events.jsonl").write_bytes(b'{"torn":')
    budget._CPU_CACHES.pop(str(tmp_path.resolve()), None)
    budget._RECONCILED_ROOTS.discard(str(tmp_path.resolve()))

    assert budget.record_v3_cpu_event(tmp_path, "optimizer:one", 1.0, category="optimizer")
    rows = (directory / "cpu_events.jsonl").read_text().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["event_id"] == "optimizer:one"


def test_concurrent_processes_cannot_append_the_same_event_twice(tmp_path):
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    queue = context.Queue()
    workers = [
        context.Process(target=_record_cpu_event_after_cache_load,
                        args=(str(tmp_path), barrier, queue))
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    results = [queue.get(timeout=15) for _ in workers]
    for worker in workers:
        worker.join(timeout=15)
        assert worker.exitcode == 0
    assert sorted(results) == [False, True]
    rows = (tmp_path / "research/g_v3/results/runtime/cpu_events.jsonl").read_text().splitlines()
    assert len(rows) == 1
    assert budget.cpu_seconds_committed(tmp_path) == 1.25


def test_identical_duplicate_cpu_events_are_repaired_once(tmp_path):
    directory = tmp_path / "research/g_v3/results/runtime"
    directory.mkdir(parents=True)
    event = {"event_id": "optimizer:duplicate", "category": "optimizer", "cpu_seconds": 0.75}
    line = json.dumps(event, sort_keys=True, separators=(",", ":"))
    (directory / "cpu_events.jsonl").write_text(line + "\n" + line + "\n")

    budget._CPU_CACHES.pop(str(tmp_path.resolve()), None)
    budget._RECONCILED_ROOTS.discard(str(tmp_path.resolve()))
    assert budget.cpu_seconds_committed(tmp_path) == 0.75
    rows = (directory / "cpu_events.jsonl").read_text().splitlines()
    assert [json.loads(row) for row in rows] == [event]
