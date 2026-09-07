"""Every class releases the GIL for work that blocks, and tolerates sharing.

The first two assert behaviour a single-threaded test cannot see: that a long
call does not pin the interpreter, and that a shared receiver tolerates
overlap. The third guards methods that already behaved correctly.
"""

import threading
import time

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
    index = vanedb.ApproxIndex(_dim(), vanedb.Metric.COSINE, capacity=4000)
    for i in range(4000):
        index.add(i, _vec(i))

    ticks = []
    stop = threading.Event()

    def tick():
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.001)

    watcher = threading.Thread(target=tick)
    watcher.start()
    saves = 10
    try:
        t0 = time.perf_counter()
        for n in range(saves):
            index.save(str(tmp_path / f"s{n}.hnsw"))
        elapsed = time.perf_counter() - t0
    finally:
        stop.set()
        watcher.join()

    # The longest the watcher went unscheduled, against how long one save
    # takes. A save that holds the GIL produces a gap the length of a whole
    # save; one that releases it keeps the watcher near its 1ms cadence. The
    # comparison is to this run's own timings, so there is no constant tuned
    # to how fast this machine happens to be.
    during = [b - a for a, b in zip(ticks, ticks[1:])]
    assert during, "watcher never ran"
    per_save = elapsed / saves
    assert max(during) < per_save / 2, (
        f"watcher stalled {max(during) * 1000:.1f}ms; one save takes "
        f"{per_save * 1000:.1f}ms, so the interpreter was held for its duration"
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
