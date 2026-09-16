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

Memory: embeddings are accumulated as contiguous float32 (≈310 MiB for the
default size), not nested Python float lists. Streaming write + hashlib avoids
a second full copy of the file.

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


def write_vnef(
    path: Path,
    dim: int,
    vectors: np.ndarray,
    queries: np.ndarray,
    ids: np.ndarray | None = None,
) -> str:
    """Write VNEF v1; return lowercase hex sha256 of the file bytes."""
    if vectors.dtype != np.float32 or queries.dtype != np.float32:
        raise ValueError("vectors/queries must be float32")
    if vectors.ndim != 2 or queries.ndim != 2:
        raise ValueError("vectors/queries must be 2-D")
    if vectors.shape[1] != dim or queries.shape[1] != dim:
        raise ValueError(f"expected dim {dim}, got {vectors.shape[1]}/{queries.shape[1]}")

    n_docs = int(vectors.shape[0])
    n_queries = int(queries.shape[0])
    if ids is None:
        ids = np.arange(n_docs, dtype=np.uint64)
    else:
        ids = np.asarray(ids, dtype=np.uint64)
        if ids.shape != (n_docs,):
            raise ValueError("ids length must equal n_docs")

    h = hashlib.sha256()
    header = MAGIC + struct.pack("<IIIIII", VERSION, dim, n_docs, n_queries, 1, 0)
    with path.open("wb") as f:
        f.write(header)
        h.update(header)
        # Contiguous C-order float32 / uint64 payloads.
        for chunk in (vectors, queries):
            blob = np.ascontiguousarray(chunk).tobytes()
            f.write(blob)
            h.update(blob)
        id_blob = np.ascontiguousarray(ids).tobytes()
        f.write(id_blob)
        h.update(id_blob)
    return h.hexdigest()


def load_nq_texts(n_docs: int, n_queries: int) -> tuple[list[str], list[str], str]:
    """Pull plain-text passages + queries from BeIR/nq parquet on Hugging Face.

    Returns (docs, queries, corpus_revision_note).
    """
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
        )
        queries_path = hf_hub_download(
            repo_id="BeIR/nq",
            filename="queries/queries-00000-of-00001.parquet",
            repo_type="dataset",
        )
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"refusing synthetic fallback: could not download BeIR/nq ({e}). "
            "Pin network access or populate HF_HUB_CACHE."
        ) from e

    corpus = pq.read_table(corpus_path, columns=["title", "text"])
    if corpus.num_rows < n_docs:
        raise SystemExit(f"corpus only had {corpus.num_rows} passages; need {n_docs}")
    titles = corpus.column("title").to_pylist()[:n_docs]
    texts = corpus.column("text").to_pylist()[:n_docs]
    docs = [f"{(t or '').strip()} {(x or '').strip()}".strip() for t, x in zip(titles, texts)]
    docs = [d for d in docs if d]
    if len(docs) < n_docs:
        raise SystemExit(f"after filtering empty rows, only {len(docs)} docs; need {n_docs}")
    docs = docs[:n_docs]

    qtab = pq.read_table(queries_path, columns=["text"])
    if qtab.num_rows < n_queries:
        raise SystemExit(f"queries only had {qtab.num_rows}; need {n_queries}")
    queries = [((q or "").strip()) for q in qtab.column("text").to_pylist()[:n_queries]]
    queries = [q for q in queries if q]
    if len(queries) < n_queries:
        raise SystemExit(f"after filtering empty queries, only {len(queries)}; need {n_queries}")
    queries = queries[:n_queries]

    note = (
        "BeIR/nq dataset parquet "
        "corpus/corpus-00000-of-00001.parquet + queries/queries-00000-of-00001.parquet"
    )
    return docs, queries, note


def embed_fastembed(texts: list[str], model: str, dim: int) -> np.ndarray:
    """Embed into a contiguous float32 matrix without nested Python floats."""
    from fastembed import TextEmbedding

    eng = TextEmbedding(model_name=model)
    out = np.empty((len(texts), dim), dtype=np.float32)
    i = 0
    # batch_size=256 + parallel=0 (all cores) for offline bulk encoding; keep
    # peak RSS in check via float32 accumulation rather than Python float lists.
    for vec in eng.embed(texts, batch_size=256, parallel=0):
        row = np.asarray(vec, dtype=np.float32)
        if row.shape != (dim,):
            raise SystemExit(f"fastembed returned dim {row.shape[0]}, expected {dim}")
        out[i] = row
        i += 1
        if i % 5000 == 0 or i == len(texts):
            print(f"  embedded {i}/{len(texts)}", flush=True)
    if i != len(texts):
        raise SystemExit(f"fastembed yielded {i} vectors, expected {len(texts)}")
    return out


def embed_ollama(texts: list[str], model: str, base: str, dim: int) -> np.ndarray:
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
        if (i + 1) % 5000 == 0 or i + 1 == len(texts):
            print(f"  embedded {i + 1}/{len(texts)}", flush=True)
    return out


def embed_openai(
    texts: list[str], model: str, base: str, api_key: str, dim: int
) -> np.ndarray:
    import urllib.request

    out = np.empty((len(texts), dim), dtype=np.float32)
    batch = 64
    filled = 0
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
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
        print(f"  embedded {filled}/{len(texts)}", flush=True)
    return out


def update_sha256sums(sums: Path, name: str, digest: str, preserve: Path | None = None) -> None:
    existing: dict[str, str] = {}
    if sums.exists():
        for line in sums.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                existing[parts[1].lstrip("*")] = parts[0]
    # Also merge smoke hash from the committed fixtures dir when regenerating elsewhere.
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
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading corpus ({args.n_docs} docs, {args.n_queries} queries)…", flush=True)
    docs, queries, corpus_note = load_nq_texts(args.n_docs, args.n_queries)

    t0 = time.time()
    print(f"embedding with {args.backend} / {args.model}…", flush=True)
    if args.backend == "fastembed":
        model = args.model
        print(f"  docs…", flush=True)
        doc_vecs = embed_fastembed(docs, model, args.dim)
        print(f"  queries…", flush=True)
        query_vecs = embed_fastembed(queries, model, args.dim)
    elif args.backend == "ollama":
        model = "nomic-embed-text" if "nomic" in args.model else args.model
        print(f"  docs…", flush=True)
        doc_vecs = embed_ollama(docs, model, args.ollama_base, args.dim)
        print(f"  queries…", flush=True)
        query_vecs = embed_ollama(queries, model, args.ollama_base, args.dim)
    else:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise SystemExit("OPENAI_API_KEY required for --backend openai")
        print(f"  docs…", flush=True)
        doc_vecs = embed_openai(docs, args.model, args.openai_base, key, args.dim)
        print(f"  queries…", flush=True)
        query_vecs = embed_openai(queries, args.model, args.openai_base, key, args.dim)
    elapsed = time.time() - t0

    dim = int(doc_vecs.shape[1])
    if dim != args.dim:
        print(f"warning: model dim {dim} != --dim {args.dim}; using {dim}", file=sys.stderr)

    out_path = args.out_dir / "embeddings.vnef"
    sha = write_vnef(out_path, dim, doc_vecs, query_vecs)
    meta = {
        "model": args.model,
        "corpus": corpus_note,
        "dim": dim,
        "n_docs": args.n_docs,
        "n_queries": args.n_queries,
        "metric_native": "cosine",
        "generator": f"scripts/generate_fixture.py --backend {args.backend}",
        "notes": f"generated in {elapsed:.1f}s; never regenerate in CI",
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    repo_sums = Path(__file__).resolve().parent.parent / "fixtures" / "SHA256SUMS"
    update_sha256sums(args.out_dir / "SHA256SUMS", "embeddings.vnef", sha, preserve=repo_sums)
    print(f"wrote {out_path} sha256={sha} size={out_path.stat().st_size}")
    print("update SHA256SUMS and host embeddings.vnef as a release asset before CI consume")


if __name__ == "__main__":
    main()
