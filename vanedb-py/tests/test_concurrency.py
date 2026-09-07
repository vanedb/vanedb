"""Every class releases the GIL for work that blocks, and tolerates sharing.

The first two assert behaviour a single-threaded test cannot see: that a long
call does not pin the interpreter, and that a shared receiver tolerates
overlap. The third guards methods that already behaved correctly.
"""

import threading
import time

import numpy as np

import vanedb


def _dim():
    return 8


def _vec(i, dim=8):
    return [float((i + j) % 17) for j in range(dim)]


def test_disk_builder_is_usable_from_two_threads(tmp_path):
    """`add` releases the GIL, so its receiver has to tolerate overlap."""
    builder = vanedb.DiskIndexBuilder(_dim(), vanedb.Metric.L2)
    errors = []

    def fill(lo, hi):
        try:
            for i in range(lo, hi):
                builder.add(i, _vec(i))
        except BaseException as exc:  # noqa: BLE001 - report, do not swallow
            errors.append(exc)

    threads = [threading.Thread(target=fill, args=(lo, lo + 250)) for lo in (0, 250)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent add raised: {errors!r}"
    assert builder.size() == 500

    path = tmp_path / "concurrent.vndb"
    builder.save(str(path))
    assert vanedb.DiskIndex.open(str(path)).size() == 500


def test_index_save_does_not_block_other_threads(tmp_path):
    """`save` serialises the whole graph; holding the GIL would freeze the process."""
    # Large enough that one save takes far longer than the watcher's 1ms
    # cadence below. At a few thousand small vectors a save lands within a
    # tick or two of that floor, and no measurement can separate "the
    # interpreter was held" from "the watcher simply had not woken yet".
    n, dim = 20_000, 64
    index = vanedb.ApproxIndex(dim, vanedb.Metric.L2, capacity=n)
    rng = np.random.default_rng(0)
    index.add_batch(
        np.arange(n, dtype=np.uint64),
        rng.random((n, dim), dtype=np.float32),
    )

    ticks = []
    stop = threading.Event()

    def tick():
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.001)

    watcher = threading.Thread(target=tick)
    watcher.start()
    try:
        started = time.perf_counter()
        index.save(str(tmp_path / "s.hnsw"))
        finished = time.perf_counter()
    finally:
        stop.set()
        watcher.join()

    # Only ticks strictly inside the save window count, so a tick that landed
    # just before the call cannot be mistaken for one during it.
    inside = [t for t in ticks if started < t < finished]
    elapsed = finished - started
    gaps = [b - a for a, b in zip([started] + inside, inside + [finished])]

    # Compared against this run's own save duration, not a constant: a save
    # that holds the GIL leaves one gap spanning the whole window.
    assert max(gaps) < elapsed / 2, (
        f"watcher stalled {max(gaps) * 1000:.1f}ms of a {elapsed * 1000:.1f}ms "
        f"save, so the interpreter was held for its duration"
    )


def test_upsert_and_remove_are_shareable(tmp_path):
    """Both take the write lock; both must be callable from several threads."""
    index = vanedb.ApproxIndex(_dim(), vanedb.Metric.L2, capacity=1000)
    for i in range(200):
        index.add(i, _vec(i))

    errors = []

    def churn(lo, hi):
        try:
            for i in range(lo, hi):
                index.upsert(i, _vec(i + 1))
                index.remove(i)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=churn, args=(lo, lo + 100)) for lo in (0, 100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent upsert/remove raised: {errors!r}"
    assert index.size() == 0
