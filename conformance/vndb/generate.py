#!/usr/bin/env python3
"""Writes the golden VNDB v1 fixtures from the specification.

Deliberately independent of both engines. A fixture produced *by* an engine
drifts with that engine, and the two engines are otherwise only ever compared
to each other -- so a coordinated layout change would pass every test. These
bytes are written from conformance/README.md's field table and nothing else.

Regenerate only when the VNDB specification itself changes.
"""

import pathlib
import struct

HERE = pathlib.Path(__file__).resolve().parent

MAGIC = b"VNDB"
VERSION = 1
DIM = 4
IDS = [10, 20, 30, 40, 50, 60]
METRICS = {"l2": 0, "cosine": 1, "dot": 2}


def vectors():
    """Deterministic, exactly representable in f32, and not symmetric — so a
    transposed or mis-strided read produces different answers, not the same."""
    return [
        [float(i * DIM + d) / 8.0 for d in range(DIM)]
        for i in range(len(IDS))
    ]


def build(metric_value: int) -> bytes:
    rows = vectors()
    out = bytearray()
    out += MAGIC                                    # 0  : 4  literal bytes
    out += struct.pack("<I", VERSION)               # 4  : 4  u32 le
    out += struct.pack("<Q", DIM)                   # 8  : 8  u64 le
    out += struct.pack("<Q", len(IDS))              # 16 : 8  u64 le
    out += struct.pack("<I", metric_value)          # 24 : 4  u32 le
    out += struct.pack("<I", 0)                     # 28 : 4  reserved, zero
    assert len(out) == 32, "header must be 32 bytes"
    for i in IDS:
        out += struct.pack("<Q", i)                 # count * u64 le
    for row in rows:
        for v in row:
            out += struct.pack("<f", v)             # count * dim * f32 le
    return bytes(out)


def main() -> None:
    for name, value in METRICS.items():
        path = HERE / f"v1_{name}.vndb"
        path.write_bytes(build(value))
        print(f"{path.name}: {path.stat().st_size} bytes, metric={value}")


if __name__ == "__main__":
    main()
