"""Independent bounds-checked VNDB parser. No engine parsing code is imported."""
import hashlib
import math
import mmap
from pathlib import Path
import struct


class Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def take(self, n):
        if n < 0 or self.pos + n > len(self.data):
            raise ValueError("truncated data")
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def num(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))[0]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_graph(data, fixture=None, expected_n=None):
    r = Reader(data)
    require(r.take(4) == b"VNDB", "magic")
    require(r.num("I") == 2, "version")
    require(r.num("I") == 1, "kind")
    metric = r.num("I")
    dim, count, capacity, m, efc, efs, seed, entry = [r.num("Q") for _ in range(8)]
    max_level, rng, rng_bytes = r.num("i"), r.num("I"), r.num("Q")
    require(metric in (0, 1, 2), "metric")
    require(dim > 0 and 0 < capacity <= 100_000_000 and count <= capacity, "dimensions/count/capacity")
    require(m >= 2 and efc > 0, "construction parameters")
    require(rng == 1 and rng_bytes == 0, "probe requires native Rust empty continuation")
    require(96 + count * (16 + dim * 4 + 8) <= len(data), "count cannot fit file")
    if expected_n is not None:
        require((dim, count, capacity, metric, m, efc, efs, seed) ==
                (768, expected_n, expected_n, 1, 16, 200, 50, 7), "probe header identity")
    if count:
        require(entry < count and 0 <= max_level <= 32, "entry/maximum level")
    else:
        require(entry == 2**64 - 1 and max_level == -1, "empty entry/level")
    starts, levels, layer_counts, degree_sums = [], [], [0] * 33, [0] * 33
    degree_histograms = [{} for _ in range(33)]
    live_ids, nodes_by_level = set(), [0] * 33
    total_links = 0
    fixture_ids_offset = 28 + (100_000 + 1000) * 768 * 4
    for slot in range(count):
        ident, level, flags = r.num("Q"), r.num("I"), r.num("I")
        require(level <= 32 and flags in (0, 1), "node level/flags")
        if not flags:
            require(ident not in live_ids, "duplicate live ID")
            live_ids.add(ident)
        vector = r.take(dim * 4)
        if fixture is not None:
            require(dim == 768 and flags == 0, "fixture node shape/flags")
            expected_id = struct.unpack_from("<Q", fixture, fixture_ids_offset + slot * 8)[0]
            require(ident == expected_id, "fixture ID mismatch")
            require(vector == fixture[28 + slot * dim * 4:28 + (slot + 1) * dim * 4], "fixture vector bits mismatch")
        else:
            require(all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", vector)), "nonfinite vector")
        starts.append(r.pos)
        levels.append(level)
        nodes_by_level[level] += 1
        for layer in range(level + 1):
            degree = r.num("Q")
            require(degree <= (2 * m if layer == 0 else m), "degree cap")
            require(degree <= max(0, count - 1), "degree exceeds possible neighbors")
            r.take(degree * 8)
            layer_counts[layer] += 1
            degree_sums[layer] += degree
            hist = degree_histograms[layer]
            hist[degree] = hist.get(degree, 0) + 1
            total_links += degree
    require(r.pos == len(data), "trailing bytes or unexpected continuation")
    if count:
        require(max(levels) == max_level and levels[entry] == max_level, "observed maximum/entry level")
    # Second pass checks edge levels using the complete first-pass level table.
    for slot, start in enumerate(starts):
        r.pos = start
        for layer in range(levels[slot] + 1):
            degree = r.num("Q")
            seen = set()
            for _ in range(degree):
                neighbor = r.num("Q")
                require(neighbor < count and neighbor != slot, "neighbor bounds/self-edge")
                require(neighbor not in seen and levels[neighbor] >= layer, "duplicate/above-level edge")
                seen.add(neighbor)
    total_layers = sum(layer_counts)
    trim = max_level + 1
    return dict(format="VNDB_v2", metric=metric, dim=dim, count=count, capacity=capacity,
                m=m, ef_construction=efc, ef_search=efs, seed=seed, entry=entry,
                max_level=max_level, continuation_encoding=rng, continuation_bytes=rng_bytes,
                live_count=len(live_ids), nodes_by_level=nodes_by_level[:trim],
                layer_node_counts=layer_counts[:trim], layer_degree_sums=degree_sums[:trim],
                layer_degree_histograms=degree_histograms[:trim], total_layers=total_layers,
                total_links=total_links, mean_degree0=degree_sums[0] / count if count else 0,
                mean_links_per_node=total_links / count if count else 0,
                exact_fit_neighbor_bytes_64bit=count * 24 + total_layers * 24 + total_links * 8,
                exact_fit_scope="n node Vec headers + one Vec header per layer + 8 bytes per edge; excludes outer stack header; assumes exact-fit capacities, NOT observed heap capacity")


def parse_disk(data, fixture, n):
    r = Reader(data)
    require(r.take(4) == b"VNDB" and r.num("I") == 1, "disk magic/version")
    dim, count, metric, reserved = r.num("Q"), r.num("Q"), r.num("I"), r.num("I")
    require((dim, count, metric, reserved) == (768, n, 1, 0), "disk header identity")
    require(len(data) == 32 + n * (8 + 768 * 4), "disk exact length")
    fixture_ids_offset = 28 + (100_000 + 1000) * 768 * 4
    require(r.take(n * 8) == fixture[fixture_ids_offset:fixture_ids_offset + n * 8], "disk ID bytes")
    require(r.take(n * dim * 4) == fixture[28:28 + n * dim * 4], "disk vector bytes")
    require(r.pos == len(data), "disk EOF")
    return dict(format="VNDB_v1", dim=dim, count=count, metric=metric, vector_bits_match=True)


def verify(path, fixture_path, kind, n):
    with Path(path).open("rb") as f, Path(fixture_path).open("rb") as source:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as fixture:
            result = parse_graph(data, fixture, n) if kind == "approx" else parse_disk(data, fixture, n)
            result.update(file=str(path), file_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                          checks="bounds/header/EOF and fixture ID/vector bytes verified")
            return result
