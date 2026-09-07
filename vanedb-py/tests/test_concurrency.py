"""Every class releases the GIL for work that blocks, and tolerates sharing.

These assert behaviour a single-threaded test cannot see: that a long call
does not pin the interpreter, and that two threads may use one object.
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
    try:
        for n in range(5):
            index.save(str(tmp_path / f"s{n}.hnsw"))
    finally:
        stop.set()
        watcher.join()

    # A GIL-holding save would stall the watcher for the whole serialisation.
    # This asserts it kept running at all, not how fast it ran.
    assert len(ticks) > 5, f"watcher thread only ran {len(ticks)} times during save"


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
