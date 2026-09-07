"""Concurrent lifecycle checks, isolated so a deadlock cannot hang pytest.

These verify coherent objects and snapshots under sharing, not scheduler timing
or a minimum number of interpreter ticks.
"""

from pathlib import Path
import subprocess
import sys
import threading

import vanedb


def _vec(i):
    return [float((i + j) % 17) for j in range(8)]


def _parallel(*operations):
    start = threading.Barrier(len(operations), timeout=10)
    errors = []

    def run(operation):
        try:
            start.wait()
            operation()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(op,), daemon=True) for op in operations]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    assert not any(thread.is_alive() for thread in threads), "worker did not finish"
    assert not errors, f"concurrent operation raised: {errors!r}"


def _disk_builder(directory):
    builder = vanedb.DiskIndexBuilder(8, vanedb.Metric.L2)

    def fill(lo):
        for i in range(lo, lo + 250):
            builder.add(i, _vec(i))

    def snapshots():
        for n in range(5):
            assert builder.dimension == 8
            assert 0 <= len(builder) <= 500
            assert 0 <= builder.size() <= 500
            path = str(directory / f"builder-{n}.vndb")
            builder.save(path)
            snapshot = vanedb.DiskIndex.open(path)
            assert snapshot.dimension == 8
            assert 0 <= snapshot.size() <= 500

    _parallel(lambda: fill(0), lambda: fill(250), snapshots)
    assert len(builder) == builder.size() == 500
    path = str(directory / "complete.vndb")
    builder.save(path)
    snapshot = vanedb.DiskIndex.open(path)
    assert snapshot.size() == 500
    for i in range(500):
        assert snapshot.get(i) == _vec(i)


def _index_snapshots(directory):
    index = vanedb.ApproxIndex(8, vanedb.Metric.COSINE, capacity=100)
    for i in range(100):
        index.add(i, _vec(i))

    def replace():
        for i in range(100):
            index.upsert(i, _vec(i + 1))

    def snapshots():
        for n in range(5):
            path = str(directory / f"graph-{n}.vndb")
            index.save(path)
            snapshot = vanedb.ApproxIndex.load(path)
            assert snapshot.size() == 100
            for i in range(100):
                assert snapshot.get_vector(i) in (_vec(i), _vec(i + 1))
            assert len(snapshot.search(_vec(0), 10)) == 10
            snapshot.add(100, _vec(100))
            assert snapshot.size() == 101

    _parallel(replace, snapshots)
    for i in range(100):
        assert index.get_vector(i) == _vec(i + 1)


def _index_churn(_directory):
    index = vanedb.ApproxIndex(8, vanedb.Metric.L2, capacity=200)
    for i in range(200):
        index.add(i, _vec(i))

    def churn(lo):
        for i in range(lo, lo + 100):
            index.upsert(i, _vec(i + 1))
            index.remove(i)

    _parallel(lambda: churn(0), lambda: churn(100))
    assert index.size() == 0


def _check_scenario(name, directory):
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), name, str(directory)],
        cwd=directory, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_disk_builder_concurrent_add_getters_and_save(tmp_path):
    _check_scenario("builder", tmp_path)


def test_index_snapshots_remain_usable_during_upsert(tmp_path):
    _check_scenario("snapshots", tmp_path)


def test_upsert_and_remove_are_shareable(tmp_path):
    _check_scenario("churn", tmp_path)


if __name__ == "__main__":
    {"builder": _disk_builder, "snapshots": _index_snapshots, "churn": _index_churn}[
        sys.argv[1]
    ](Path(sys.argv[2]))
