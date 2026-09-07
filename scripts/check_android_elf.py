#!/usr/bin/env python3
"""Reject Android ELF64 artifacts that cannot load with 16 KiB memory pages."""

from pathlib import Path
import struct
import sys


def check(path: Path) -> None:
    data = path.read_bytes()
    if len(data) < 64 or data[:6] != b"\x7fELF\x02\x01":
        raise ValueError(f"{path}: expected a little-endian ELF64 binary")
    offset = struct.unpack_from("<Q", data, 32)[0]
    entry_size, count = struct.unpack_from("<HH", data, 54)
    if entry_size != 56 or offset + count * entry_size > len(data):
        raise ValueError(f"{path}: invalid program-header table")
    loads = 0
    for index in range(count):
        kind, _, file_offset, address, _, _, _, alignment = struct.unpack_from(
            "<IIQQQQQQ", data, offset + index * entry_size
        )
        if kind != 1:  # PT_LOAD
            continue
        loads += 1
        if alignment < 16384 or alignment & (alignment - 1) or (address - file_offset) % 16384:
            raise ValueError(f"{path}: LOAD segment {index} is not 16 KiB aligned")
    if not loads:
        raise ValueError(f"{path}: no LOAD segments")
    print(f"{path}: {loads} LOAD segments support 16 KiB pages")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: check_android_elf.py BINARY [BINARY ...]")
    try:
        for argument in sys.argv[1:]:
            check(Path(argument))
    except (OSError, ValueError) as error:
        sys.exit(str(error))
