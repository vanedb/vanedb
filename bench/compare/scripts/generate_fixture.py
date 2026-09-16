#!/usr/bin/env python3
"""Generate the real embedding fixture for bench/compare (RFC 0003 / #198).

Produces embeddings.vnef + metadata.json. Never run this in CI — generate once,
checksum, and host the bytes (release asset or internal cache). The harness only
loads and verifies.

Default: nomic-embed-text-v1.5 (768-d) over a slice of the Hugging Face
`BeIR/nq` corpus (natural questions passages), 100k docs / 1k queries.

Dependencies (install in a venv):
  pip install numpy huggingface_hub pyarrow httpx

Embedding backends (pick one):
  --backend fastembed   requires: pip install fastembed
  --backend ollama      requires a local Ollama with `ollama pull nomic-embed-text`
  --backend openai      OPENAI_API_KEY + any OpenAI-compatible /v1/embeddings host

Memory: streams BeIR/nq parquet in text batches and appends float32 rows to the
VNEF file while hashing. Peak RSS stays near model + one batch (not 100k×768
Python floats). Set VANEDB_FIXTURE_BATCH (default 512) if a host still OOMs.

The VNEF writer is duplicated in Rust (`fixture.rs`); keep the header layout
in lockstep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

MAGIC = b"VNEF"
VERSION = 1
# Pin BeIR/nq so regenerations are byte-stable across hosts.
BEIR_NQ_REVISION = "b7253e6c379163d024ddb1d6948152a91a2e3b46"
# nomic-embed-text-v1.5 asymmetric task prefixes (required for faithful embeddings).
NOMIC_DOC_PREFIX = "search_document: "
NOMIC_QUERY_PREFIX = "search_query: "


class VnefWriter:
    """Stream VNEF v1 to disk: header first, then docs, queries, ids."""

    def __init__(self, path: Path, dim: int, n_docs: int, n_queries: int) -> None:
        self.path = path
        self.dim = dim
        self.n_docs = n_docs
        self.n_queries = n_queries
        self.h = hashlib.sha256()
        self._docs_written = 0
        self._queries_written = 0
        self._f = path.open("wb")
        header = MAGIC + struct.pack("<IIIIII", VERSION, dim, n_docs, n_queries, 1, 0)
        self._write(header)

    def _write(self, blob: bytes) -> None:
        self._f.write(blob)
        self.h.update(blob)

    def write_vectors(self, mat: np.ndarray, *, kind: str) -> None:
        if mat.dtype != np.float32 or mat.ndim != 2 or mat.shape[1] != self.dim:
            raise ValueError(f"expected float32 (*, {self.dim}), got {mat.dtype} {mat.shape}")
        blob = np.ascontiguousarray(mat).tobytes()
        self._write(blob)
        if kind == "docs":
            self._docs_written += mat.shape[0]
        elif kind == "queries":
            self._queries_written += mat.shape[0]
        else:
            raise ValueError(kind)

    def finish(self, ids: np.ndarray | None = None) -> str:
        if self._docs_written != self.n_docs:
            raise SystemExit(f"wrote {self._docs_written} docs, expected {self.n_docs}")
        if self._queries_written != self.n_queries:
            raise SystemExit(f"wrote {self._queries_written} queries, expected {self.n_queries}")
        if ids is None:
            ids = np.arange(self.n_docs, dtype=np.uint64)
        else:
            ids = np.asarray(ids, dtype=np.uint64)
            if ids.shape != (self.n_docs,):
                raise ValueError("ids length must equal n_docs")
        self._write(np.ascontiguousarray(ids).tobytes())
        self._f.close()
        return self.h.hexdigest()


def open_nq_parquet():
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub and pyarrow are required: pip install huggingface_hub pyarrow"
        ) from e

    try:
        corpus_path = hf_hub_download(
            repo_id="BeIR/nq",
            filename="corpus/corpus-00000-of-00001.parquet",
            repo_type="dataset",
            revision=BEIR_NQ_REVISION,
        )
        queries_path = hf_hub_download(
            repo_id="BeIR/nq",
            filename="queries/queries-00000-of-00001.parquet",
            repo_type="dataset",
            revision=BEIR_NQ_REVISION,
        )
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"refusing synthetic fallback: could not download BeIR/nq@{BEIR_NQ_REVISION} ({e}). "
            "Pin network access or populate HF_HUB_CACHE."
        ) from e

    note = (
        f"BeIR/nq dataset revision {BEIR_NQ_REVISION} "
        "corpus/corpus-00000-of-00001.parquet + queries/queries-00000-of-00001.parquet"
    )
    return pq, corpus_path, queries_path, note


def iter_nq_docs(pq, corpus_path: str, n_docs: int, batch: int, max_chars: int = 0):
    """Yield non-empty doc strings until n_docs collected; stream parquet batches."""
    pf = pq.ParquetFile(corpus_path)
    collected = 0
    buf: list[str] = []
    for batch_table in pf.iter_batches(columns=["title", "text"], batch_size=max(batch * 2, 1024)):
        titles = batch_table.column("title").to_pylist()
        texts = batch_table.column("text").to_pylist()
        for t, x in zip(titles, texts):
            if collected >= n_docs:
                break
            doc = f"{(t or '').strip()} {(x or '').strip()}".strip()
            if not doc:
                continue
            if max_chars:
                doc = truncate_text(doc, max_chars)
            buf.append(doc)
            collected += 1
            if len(buf) >= batch:
                yield buf
                buf = []
        if collected >= n_docs:
            break
    if buf:
        yield buf
    if collected < n_docs:
        raise SystemExit(f"after filtering empty rows, only {collected} docs; need {n_docs}")


def load_nq_queries(pq, queries_path: str, n_queries: int, max_chars: int = 0) -> list[str]:
    qtab = pq.read_table(queries_path, columns=["text"])
    if qtab.num_rows < n_queries:
        raise SystemExit(f"queries only had {qtab.num_rows}; need {n_queries}")
    queries = [((q or "").strip()) for q in qtab.column("text").to_pylist()[: n_queries * 2]]
    queries = [q for q in queries if q][:n_queries]
    if max_chars:
        queries = [truncate_text(q, max_chars) for q in queries]
    if len(queries) < n_queries:
        raise SystemExit(f"after filtering empty queries, only {len(queries)}; need {n_queries}")
    return queries


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    # Prefer a word boundary so we don't feed truncated mid-token junk.
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut or text[:max_chars]


def make_fastembed(model: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model)


def embed_batch_fastembed(eng, texts: list[str], dim: int) -> np.ndarray:
    import gc

    out = np.empty((len(texts), dim), dtype=np.float32)
    i = 0
    # Tiny ORT batches: BeIR passages are long; ORT arena grows across batches
    # on this host unless we keep pressure low and GC between text batches.
    for vec in eng.embed(texts, batch_size=min(8, len(texts))):
        row = np.asarray(vec, dtype=np.float32)
        if row.shape != (dim,):
            raise SystemExit(f"fastembed returned dim {row.shape[0]}, expected {dim}")
        out[i] = row
        i += 1
    if i != len(texts):
        raise SystemExit(f"fastembed yielded {i} vectors, expected {len(texts)}")
    gc.collect()
    return out


def embed_batch_ollama(texts: list[str], model: str, base: str, dim: int) -> np.ndarray:
    import urllib.request

    out = np.empty((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        req = urllib.request.Request(
            f"{base.rstrip('/')}/api/embeddings",
            data=json.dumps({"model": model, "prompt": t}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode())
        row = np.asarray(payload["embedding"], dtype=np.float32)
        if row.shape != (dim,):
            raise SystemExit(f"ollama returned dim {row.shape[0]}, expected {dim}")
        out[i] = row
    return out


def embed_batch_openai(
    texts: list[str], model: str, base: str, api_key: str, dim: int
) -> np.ndarray:
    import urllib.request

    out = np.empty((len(texts), dim), dtype=np.float32)
    filled = 0
    step = 64
    for i in range(0, len(texts), step):
        chunk = texts[i : i + step]
        body = json.dumps({"model": model, "input": chunk}).encode()
        req = urllib.request.Request(
            f"{base.rstrip('/')}/v1/embeddings",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode())
        data = sorted(payload["data"], key=lambda d: d["index"])
        for d in data:
            row = np.asarray(d["embedding"], dtype=np.float32)
            if row.shape != (dim,):
                raise SystemExit(f"openai returned dim {row.shape[0]}, expected {dim}")
            out[filled] = row
            filled += 1
    return out


def update_sha256sums(sums: Path, name: str, digest: str, preserve: Path | None = None) -> None:
    existing: dict[str, str] = {}
    if sums.exists():
        for line in sums.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                existing[parts[1].lstrip("*")] = parts[0]
    if preserve is not None and preserve.exists():
        for line in preserve.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].lstrip("*") == "smoke.vnef":
                existing.setdefault("smoke.vnef", parts[0])
    existing[name] = digest
    smoke_path = sums.parent / "smoke.vnef"
    if "smoke.vnef" not in existing and smoke_path.exists():
        existing["smoke.vnef"] = hashlib.sha256(smoke_path.read_bytes()).hexdigest()
    sums.write_text("".join(f"{h}  {n}\n" for n, h in sorted(existing.items())))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("fixtures"))
    ap.add_argument("--n-docs", type=int, default=100_000)
    ap.add_argument("--n-queries", type=int, default=1_000)
    ap.add_argument("--backend", choices=["fastembed", "ollama", "openai"], default="fastembed")
    ap.add_argument(
        "--model",
        default="nomic-ai/nomic-embed-text-v1.5",
        help="Model id for the chosen backend",
    )
    ap.add_argument("--ollama-base", default="http://127.0.0.1:11434")
    ap.add_argument("--openai-base", default="https://api.openai.com")
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument(
        "--text-batch",
        type=int,
        default=int(os.environ.get("VANEDB_FIXTURE_BATCH", "64")),
        help="How many passages to hold/embed at once (lower if OOM)",
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=int(os.environ.get("VANEDB_FIXTURE_MAX_CHARS", "2000")),
        help="Truncate each passage/query to this many chars (0=disable). "
        "Cuts ORT peak RSS on long BeIR NQ documents.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"streaming corpus ({args.n_docs} docs, {args.n_queries} queries, "
        f"text_batch={args.text_batch}, max_chars={args.max_chars})…",
        flush=True,
    )
    pq, corpus_path, queries_path, corpus_note = open_nq_parquet()
    queries = load_nq_queries(pq, queries_path, args.n_queries, args.max_chars)

    t0 = time.time()
    print(f"embedding with {args.backend} / {args.model}…", flush=True)
    out_path = args.out_dir / "embeddings.vnef"
    writer = VnefWriter(out_path, args.dim, args.n_docs, args.n_queries)

    if args.backend == "fastembed":
        eng = make_fastembed(args.model)

        def embed_docs(texts: list[str]) -> np.ndarray:
            return embed_batch_fastembed(eng, texts, args.dim)

        def embed_queries(texts: list[str]) -> np.ndarray:
            return embed_batch_fastembed(eng, texts, args.dim)
    elif args.backend == "ollama":
        model = "nomic-embed-text" if "nomic" in args.model else args.model

        def embed_docs(texts: list[str]) -> np.ndarray:
            return embed_batch_ollama(texts, model, args.ollama_base, args.dim)

        def embed_queries(texts: list[str]) -> np.ndarray:
            return embed_batch_ollama(texts, model, args.ollama_base, args.dim)
    else:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise SystemExit("OPENAI_API_KEY required for --backend openai")

        def embed_docs(texts: list[str]) -> np.ndarray:
            return embed_batch_openai(texts, args.model, args.openai_base, key, args.dim)

        def embed_queries(texts: list[str]) -> np.ndarray:
            return embed_batch_openai(texts, args.model, args.openai_base, key, args.dim)

    done = 0
    use_nomic = "nomic" in args.model.lower()
    if use_nomic:
        print(
            f"  applying nomic task prefixes ({NOMIC_DOC_PREFIX!r} / {NOMIC_QUERY_PREFIX!r})",
            flush=True,
        )
    print("  docs…", flush=True)
    for batch_texts in iter_nq_docs(
        pq, corpus_path, args.n_docs, args.text_batch, args.max_chars
    ):
        if use_nomic:
            batch_texts = [NOMIC_DOC_PREFIX + t for t in batch_texts]
        mat = embed_docs(batch_texts)
        writer.write_vectors(mat, kind="docs")
        done += mat.shape[0]
        rate = done / max(time.time() - t0, 1e-6)
        print(f"  embedded docs {done}/{args.n_docs} ({rate:.1f} vec/s)", flush=True)
        # Recreate the ONNX session periodically — arena growth otherwise
        # climbs toward OOM across a 100k run on 16 GiB hosts.
        if args.backend == "fastembed" and done % (args.text_batch * 20) == 0:
            import gc

            eng = make_fastembed(args.model)

            def embed_docs(texts: list[str], _eng=eng) -> np.ndarray:
                return embed_batch_fastembed(_eng, texts, args.dim)

            def embed_queries(texts: list[str], _eng=eng) -> np.ndarray:
                return embed_batch_fastembed(_eng, texts, args.dim)

            gc.collect()

    print("  queries…", flush=True)
    # Queries are small; still batch for API backends.
    q_batch = args.text_batch
    for i in range(0, len(queries), q_batch):
        chunk = queries[i : i + q_batch]
        if use_nomic:
            chunk = [NOMIC_QUERY_PREFIX + t for t in chunk]
        mat = embed_queries(chunk)
        writer.write_vectors(mat, kind="queries")
        print(f"  embedded queries {min(i + q_batch, len(queries))}/{args.n_queries}", flush=True)

    sha = writer.finish()
    elapsed = time.time() - t0
    meta = {
        "model": args.model,
        "corpus": corpus_note,
        "dim": args.dim,
        "n_docs": args.n_docs,
        "n_queries": args.n_queries,
        "metric_native": "cosine",
        "generator": (
            f"scripts/generate_fixture.py --backend {args.backend} "
            f"--text-batch {args.text_batch} --max-chars {args.max_chars}"
        ),
        "notes": (
            f"generated in {elapsed:.1f}s; max_chars={args.max_chars}; "
            f"beir_nq_revision={BEIR_NQ_REVISION}; "
            f"nomic_prefixes={'yes' if use_nomic else 'no'}; "
            "never regenerate in CI"
        ),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    repo_sums = Path(__file__).resolve().parent.parent / "fixtures" / "SHA256SUMS"
    update_sha256sums(args.out_dir / "SHA256SUMS", "embeddings.vnef", sha, preserve=repo_sums)
    print(f"wrote {out_path} sha256={sha} size={out_path.stat().st_size}")
    print("update SHA256SUMS and host embeddings.vnef as a release asset before CI consume")


if __name__ == "__main__":
    main()
