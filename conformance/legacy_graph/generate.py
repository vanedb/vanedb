#!/usr/bin/env python3
"""Encode fixed pre-1.0 graph fixtures independently of either engine's writer.

Do not regenerate these to accommodate a loader change. Their old layouts are
the compatibility contract; add a new fixture for a newly supported layout.
"""

import hashlib
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]
IDS = [101, 202, (1 << 64) - 1]
VECTORS = [1.0, 0.0, 0.0, 1.0, 0.8, 0.2]
LEVELS = [1, 0, 1]
NEIGHBORS = [[[1, 2], [2]], [[0, 2]], [[0, 1], [0]]]


class Payload:
    def __init__(self):
        self.bytes = bytearray()

    def put(self, fmt, *values):
        self.bytes.extend(struct.pack("<" + fmt, *values))

    def vector(self, fmt, values):
        self.put("Q", len(values))
        self.put(fmt * len(values), *values)

    def graph(self, padding):
        self.put("Q", len(NEIGHBORS) + padding)
        for layers in NEIGHBORS + [[]] * padding:
            self.put("Q", len(layers))
            for neighbors in layers:
                self.vector("Q", neighbors)


def rust(version, metric, deleted=False):
    p = Payload()
    padding = 1 if version == 1 else 0
    ids = IDS.copy()
    if deleted:
        ids[1] = ids[0]  # A deleted slot whose ID belongs to another live slot.
    p.put("4sI", b"HNSW", version)
    p.put("QIQQQQQQdQQBQi", 2, metric, 4, 2, 2, 4, 16, 16,
          1 / math.log(2), 42, 3, 1, 0, 1)
    p.vector("f", VECTORS + [0.0, 0.0] * padding)
    p.vector("Q", ids + [0] * padding)
    p.vector("i", LEVELS + [0] * padding)
    p.graph(padding)
    mapping = [(identifier, slot) for slot, identifier in enumerate(ids)
               if not (deleted and slot == 1)]
    p.put("Q", len(mapping))
    for identifier, slot in mapping:
        p.put("QQ", identifier, slot)
    return p.bytes


def main():
    fixtures = {}
    for version in [1, 2]:
        for metric, name in enumerate(["l2", "cosine", "dot"]):
            fixtures[f"vanedb/tests/fixtures/legacy_graph/v{version}_{name}.hnsw"] = rust(version, metric)
    fixtures["vanedb/tests/fixtures/legacy_graph/v2_deleted_id_reuse.hnsw"] = rust(2, 1, deleted=True)
    hashes = []
    for name, content in sorted(fixtures.items()):
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        hashes.append(f"{hashlib.sha256(content).hexdigest()}  {name}\n")
    Path(__file__).with_name("SHA256SUMS").write_text("".join(hashes))


if __name__ == "__main__":
    main()
