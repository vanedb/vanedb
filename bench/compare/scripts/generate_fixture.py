#!/usr/bin/env python3
"""Generate the real embedding fixture for bench/compare (RFC 0003 / #198).

Produces embeddings.vnef + metadata.json. Never run this in CI — generate once,
checksum, and host the bytes (release asset or internal cache). The harness only
loads and verifies.

Default: nomic-embed-text-v1.5 (768-d) over a slice of the Hugging Face
`BeIR/nq` corpus (natural questions passages), 100k docs / 1k queries.

Dependencies (install in a venv):
  pip install numpy huggingface_hub httpx

Embedding backends (pick one):
  --backend fastembed   requires: pip install fastembed
  --backend ollama      requires a local Ollama with `ollama pull nomic-embed-text`
  --backend openai      OPENAI_API_KEY + any OpenAI-compatible /v1/embeddings host

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

MAGIC = b"VNEF"
VERSION = 1


def write_vnef(
    path: Path,
    dim: int,
    vectors: list[list[float]],
    queries: list[list[float]],
    ids: list[int] | None = None,
) -> str:
    n_docs = len(vectors)
    n_queries = len(queries)
    if ids is None:
        ids = list(range(n_docs))
    assert len(ids) == n_docs
    assert all(len(v) == dim for v in vectors)
    assert all(len(q) == dim for q in queries)

    parts: list[bytes] = [
        MAGIC,
        struct.pack("<IIIIII", VERSION, dim, n_docs, n_queries, 1, 0),
    ]
    for row in vectors:
        parts.append(struct.pack(f"<{dim}f", *row))
    for row in queries:
        parts.append(struct.pack(f"<{dim}f", *row))
    for i in ids:
        parts.append(struct.pack("<Q", i))
    blob = b"".join(parts)
    path.write_bytes(blob)
    return hashlib.sha256(blob).hexdigest()


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


def embed_fastembed(texts: list[str], model: str) -> list[list[float]]:
    from fastembed import TextEmbedding

    eng = TextEmbedding(model_name=model)
    out: list[list[float]] = []
    for vec in eng.embed(texts, batch_size=64):
        out.append([float(x) for x in vec])
    return out


def embed_ollama(texts: list[str], model: str, base: str) -> list[list[float]]:
    import urllib.request

    out: list[list[float]] = []
    for t in texts:
        req = urllib.request.Request(
            f"{base.rstrip('/')}/api/embeddings",
            data=json.dumps({"model": model, "prompt": t}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode())
        out.append([float(x) for x in payload["embedding"]])
    return out


def embed_openai(texts: list[str], model: str, base: str, api_key: str) -> list[list[float]]:
    import urllib.request

    out: list[list[float]] = []
    batch = 64
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
        # Preserve input order.
        data = sorted(payload["data"], key=lambda d: d["index"])
        for d in data:
            out.append([float(x) for x in d["embedding"]])
    return out


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
        # fastembed's nomic v1.5 id:
        model = args.model
        if model == "nomic-ai/nomic-embed-text-v1.5":
            model = "nomic-ai/nomic-embed-text-v1.5"
        doc_vecs = embed_fastembed(docs, model)
        query_vecs = embed_fastembed(queries, model)
    elif args.backend == "ollama":
        model = "nomic-embed-text" if "nomic" in args.model else args.model
        doc_vecs = embed_ollama(docs, model, args.ollama_base)
        query_vecs = embed_ollama(queries, model, args.ollama_base)
    else:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise SystemExit("OPENAI_API_KEY required for --backend openai")
        doc_vecs = embed_openai(docs, args.model, args.openai_base, key)
        query_vecs = embed_openai(queries, args.model, args.openai_base, key)
    elapsed = time.time() - t0

    dim = len(doc_vecs[0])
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
    sums = args.out_dir / "SHA256SUMS"
    # Preserve smoke entry if present.
    lines = []
    if sums.exists():
        for line in sums.read_text().splitlines():
            if line.strip() and "embeddings.vnef" not in line and "smoke.vnef" in line:
                lines.append(line)
            elif line.strip() and "smoke.vnef" in line:
                lines.append(line)
    # Always rewrite both known entries we care about.
    existing = {}
    if sums.exists():
        for line in sums.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                existing[parts[1].lstrip("*")] = parts[0]
    existing["embeddings.vnef"] = sha
    if "smoke.vnef" not in existing and (args.out_dir / "smoke.vnef").exists():
        existing["smoke.vnef"] = hashlib.sha256(
            (args.out_dir / "smoke.vnef").read_bytes()
        ).hexdigest()
    sums.write_text("".join(f"{h}  {n}\n" for n, h in sorted(existing.items())) )
    print(f"wrote {out_path} sha256={sha}")
    print(f"update SHA256SUMS and host embeddings.vnef as a release asset before CI consume")


if __name__ == "__main__":
    main()
