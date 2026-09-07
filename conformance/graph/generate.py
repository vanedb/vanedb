#!/usr/bin/env python3
"""Encode the candidate VNDB graph field table without using either engine."""

import hashlib
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEST = ROOT / "vanedb/tests/fixtures/vndb_graph"


def rng_state(kind):
    if kind == 1:
        return b""
    state = [42]
    for i in range(1, 624):
        state.append((1812433253 * (state[-1] ^ (state[-1] >> 30)) + i) & 0xFFFFFFFF)
    return (" ".join(map(str, state)) + (" 624" if kind == 2 else "")).encode("ascii")


def encode(metric, kind=1, deleted=False, empty=False, dead_slots=()):
    ids = [101, 101 if deleted else 202, (1 << 64) - 1]
    rows = [[1, 0], [0, 1], [0.8, 0.2]]
    levels = [1, 0, 1]
    graph = [[[1, 2], [2]], [[0, 2]], [[0, 1], [0]]]
    state = rng_state(kind)
    data = bytearray(struct.pack("<4sIIIQQQQQQQQiIQ", b"VNDB", 2, 1, metric,
        2, 0 if empty else 3, 4, 2, 16, 16, 42, (1 << 64) - 1 if empty else 0,
        -1 if empty else 1, kind, len(state)))
    assert len(data) == 96
    if not empty:
        for slot in range(3):
            data.extend(struct.pack("<QIIff", ids[slot], levels[slot], int((deleted and slot == 1) or slot in dead_slots), *rows[slot]))
            for layer in graph[slot]:
                data.extend(struct.pack("<Q" + "Q" * len(layer), len(layer), *layer))
    data.extend(state)
    return data


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    files = {f"{name}_rng{kind}.vndb": encode(metric, kind)
             for metric, name in enumerate(["l2", "cosine", "dot"]) for kind in [1, 2, 3]}
    files["deleted_id_reuse.vndb"] = encode(1, deleted=True)
    files["deleted_entry.vndb"] = encode(0, dead_slots=(0,))
    files["all_deleted.vndb"] = encode(0, dead_slots=(0, 1, 2))
    files["empty.vndb"] = encode(0, empty=True)
    hashes = []
    for name, data in sorted(files.items()):
        path = DEST / name
        path.write_bytes(data)
        hashes.append(f"{hashlib.sha256(data).hexdigest()}  {path.relative_to(ROOT)}\n")
    (HERE / "SHA256SUMS").write_text("".join(hashes))


if __name__ == "__main__":
    main()
