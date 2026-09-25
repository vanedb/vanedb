"""Filtered search across the installed wheel's three index implementations."""

import subprocess
import sys
import textwrap

import pytest
import vanedb


@pytest.fixture(params=["flat", "approx", "disk"])
def index(request, tmp_path):
    if request.param == "disk":
        builder = vanedb.DiskIndexBuilder(1)
        for id, value in zip([0, 1, 2, 2**64 - 1], [0., 1., 2., 3.]):
            builder.add(id, [value])
        path = tmp_path / "filtered.vndb"
        builder.save(path)
        return vanedb.DiskIndex.open(path)
    cls = vanedb.FlatIndex if request.param == "flat" else vanedb.ApproxIndex
    index = cls(1)
    index.add_batch([0, 1, 2, 2**64 - 1], [[0.], [1.], [2.], [3.]])
    return index


def test_empty_and_full_width_id_filters(index):
    assert index.search([0.], 4, allow_ids=[]) == []
    assert index.search([0.], 4, deny_ids=[]) == index.search([0.], 4)
    assert index.search([0.], 4, allow_ids=[2**64 - 1]) == [(2**64 - 1, 9.)]
    assert [id for id, _ in index.search([0.], 4, filter=lambda id: id > 1)] == [2, 2**64 - 1]


@pytest.mark.parametrize("field", ["allow_ids", "deny_ids"])
@pytest.mark.parametrize("invalid", [[-1], [2**64], [2, 1], [1, 1]])
def test_filter_list_validation(index, field, invalid):
    with pytest.raises(ValueError):
        index.search([0.], 4, **{field: invalid})


@pytest.mark.parametrize("truth_conversion", [False, True])
def test_predicate_errors_propagate_original_exception(index, truth_conversion):
    error = RuntimeError("metadata lookup failed")
    calls = []

    class BadTruth:
        def __bool__(self):
            raise error

    def predicate(id):
        calls.append(id)
        if truth_conversion:
            return BadTruth()
        raise error

    with pytest.raises(RuntimeError) as raised:
        index.search([0.], 4, filter=predicate)
    assert raised.value is error
    assert len(calls) == 1, "stop running user code after the first failure"
    assert len(index.search([0.], 4)) == 4, "failed callbacks leave the index usable"


def test_approx_predicate_can_search_another_index():
    index = vanedb.ApproxIndex(1)
    other = vanedb.ApproxIndex(1)
    index.add(1, [0.])
    other.add(2, [0.])
    assert index.search([0.], 1, filter=lambda _: bool(other.search([0.], 1))) == [(1, 0.)]


@pytest.mark.parametrize("index_class", ["FlatIndex", "ApproxIndex"])
def test_predicate_search_does_not_hold_gil_while_waiting_for_core_lock(index_class):
    # A callback pauses with a read lock, while a writer queues. Another
    # predicate search must release the GIL while waiting behind that writer,
    # so the first callback can resume and release its read lock.
    script = textwrap.dedent(f"""
        import threading
        import time
        import vanedb
        index = vanedb.{index_class}(1)
        index.add(1, [0.])
        entered = threading.Event()
        release = threading.Event()
        errors = []
        def predicate(_):
            entered.set()
            assert release.wait(5)
            return True
        def run(fn):
            try:
                fn()
            except BaseException as error:
                errors.append(error)
        first = threading.Thread(target=lambda: run(lambda: index.search([0.], 1, filter=predicate)))
        first.start()
        assert entered.wait(5)
        writer = threading.Thread(target=lambda: run(lambda: index.add(2, [1.])))
        writer.start()
        time.sleep(0.1)
        second = threading.Thread(target=lambda: run(lambda: index.search([0.], 1, filter=lambda _: True)))
        second.start()
        time.sleep(0.1)
        release.set()
        for thread in [first, writer, second]:
            thread.join(5)
            assert not thread.is_alive()
        assert not errors, errors
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("reentry", ["search", "len", "contains", "get"])
def test_predicate_calling_back_into_the_same_index_raises(index, reentry):
    # The predicate runs under the index's read lock. A write from inside it
    # would wait on that lock forever, and a read would too once another
    # thread queues a writer; both used to hang silently. DiskIndex has no
    # lock, but the rule is the same for every index so a program can swap
    # index types without changing its predicates.
    calls = []

    def predicate(id):
        calls.append(id)
        if reentry == "search":
            index.search([0.], 1)
        elif reentry == "len":
            len(index)
        elif reentry == "contains":
            index.contains(0)
        else:
            index.get(0)
        return True

    with pytest.raises(RuntimeError, match="must not call methods on the index being searched"):
        index.search([0.], 4, filter=predicate)
    assert len(calls) == 1, "stop running user code after the first failure"
    # The guard is released with the failed search.
    assert len(index.search([0.], 4)) == 4
    assert len(index.search([0.], 4, filter=lambda _: True)) == 4
    assert len(index) == 4


@pytest.mark.parametrize("index_class", ["FlatIndex", "ApproxIndex"])
def test_predicate_writing_to_the_same_index_raises(index_class):
    index = getattr(vanedb, index_class)(1)
    index.add(1, [0.])
    with pytest.raises(RuntimeError, match="must not call methods"):
        index.search([0.], 1, filter=lambda _: index.add(2, [1.]) or True)
    index.add(2, [1.])
    assert len(index) == 2


def test_reentry_guard_is_per_index_and_per_thread():
    import threading

    index = vanedb.ApproxIndex(1)
    other = vanedb.ApproxIndex(1)
    index.add(1, [0.])
    other.add(2, [0.])
    # A different index stays usable from the predicate...
    assert index.search([0.], 1, filter=lambda _: other.contains(2)) == [(1, 0.)]
    # ...and so does this index from another thread, which merely queues.
    outcomes = []

    def predicate(_):
        worker = threading.Thread(target=lambda: outcomes.append(len(index)))
        worker.start()
        worker.join(5)
        return True

    assert index.search([0.], 1, filter=predicate) == [(1, 0.)]
    assert outcomes == [1]


def test_selective_search_widens_after_visits_exceed_beam_cap():
    # Issue #251: the first width-10 pass visits >200 nodes, but must still
    # widen to its beam cap when fewer than k matches have been accepted.
    import numpy as np

    rng = np.random.default_rng(20260920)
    vectors = rng.normal(size=(2000, 16)).astype("float32")
    index = vanedb.ApproxIndex(16, vanedb.Metric.L2, capacity=2000, seed=42)
    index.add_batch(list(range(2000)), vectors)
    query = rng.normal(size=16).astype("float32")
    allowed = list(range(0, 2000, 100))
    calls = []

    def predicate(id):
        calls.append(id)
        return id in allowed

    narrow = index.search(query, 10, ef_search=10, max_ef_search=10, filter=predicate)
    assert len(narrow) < 10
    assert len(calls) > 200
    assert index.search(query, 10, ef_search=10, max_ef_search=10,
                        allow_ids=allowed) == narrow
    for options in ({}, {"max_ef_search": 40}, {"max_ef_search": 100},
                    {"max_ef_search": 200}):
        calls.clear()
        hits = index.search(query, 10, ef_search=10, filter=predicate, **options)
        assert len(hits) == 10
        assert all(id in allowed for id, _ in hits)
        assert len(calls) > len(set(calls)), "retries revisit nodes"
        assert index.search(query, 10, ef_search=10, allow_ids=allowed, **options) == hits
