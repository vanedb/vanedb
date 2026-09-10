# Getting started: from text to search results

VaneDB stores vectors — lists of numbers — and finds the stored ones nearest to
a query. It does not turn your text into those numbers. That job belongs to an
**embedding model**, and you have to pick one. This page picks one for you, gets
it running, and ends with a script that searches five sentences.

You need Python 3.11+ and about ten minutes. No account is required for the
recommended path.

```
your text ──► embedding model ──► [0.12, -0.44, ...] ──► VaneDB ──► nearest ids
              (this page)                                (the database)
```

## 1. Pick a provider

| Option | Runs where | Cost | Setup | Suggested model | Dimensions |
|---|---|---|---|---|---|
| **Ollama** | your machine, background server | free | install app, pull model | `nomic-embed-text` | 768 |
| **sentence-transformers** | inside your Python process | free | one `pip install` | `all-MiniLM-L6-v2` | 384 |
| **OpenAI** | their servers | paid per token | API key | `text-embedding-3-small` | 1536 |

If you have no opinion: use **Ollama**. It is free, your text never leaves the
machine, and the same setup serves every project afterwards. Choose
sentence-transformers instead if you want no server at all and only a `pip
install`; choose OpenAI if you want the strongest quality with nothing running
locally and accept sending your text to a third party.

Other providers work the same way — Cohere, Voyage AI, Google Gemini, Mistral,
Jina, and any model on Hugging Face. All of them take text and return a list of
floats. Swapping providers means rewriting one function, `embed()`, in the
script below.

## 2. Set up the one you picked

### Ollama

Install it (macOS/Linux; on Windows download the installer from
<https://ollama.com/download>):

```sh
curl -fsSL https://ollama.com/install.sh | sh
ollama pull nomic-embed-text
```

Check that it answers — this should print a long list of numbers:

```sh
curl http://localhost:11434/api/embed \
  -d '{"model":"nomic-embed-text","input":"hello"}'
```

The `embed()` function, using only the standard library:

```python
import json
import urllib.request

def embed(texts):
    """texts: list[str] -> list[list[float]]"""
    body = json.dumps({"model": "nomic-embed-text", "input": texts}).encode()
    request = urllib.request.Request(
        "http://localhost:11434/api/embed",
        body,
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)["embeddings"]
```

Larger, slower alternative on the same setup: `ollama pull mxbai-embed-large`
(1024 dimensions).

### sentence-transformers

```sh
python -m pip install sentence-transformers
```

The first call downloads the model (about 90 MB) and is slow; later calls are
not.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts):
    """texts: list[str] -> list[list[float]]"""
    return model.encode(texts).tolist()
```

### OpenAI

Create a key at <https://platform.openai.com/api-keys>, then:

```sh
python -m pip install openai
export OPENAI_API_KEY="sk-..."        # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
```

```python
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from the environment

def embed(texts):
    """texts: list[str] -> list[list[float]]"""
    response = client.embeddings.create(
        model="text-embedding-3-small", input=texts
    )
    return [item.embedding for item in response.data]
```

Never paste the key into a file you commit. Your text is sent to OpenAI and
billed per token.

## 3. Search something

Install VaneDB, if you have not already:

```sh
python -m pip install vanedb
```

From a checkout of this repository instead, which needs a Rust toolchain to
build:

```sh
python -m pip install ./vanedb-py
```

Save this as `search.py`, with your `embed()` from step 2 pasted where marked,
and run `python search.py`.

```python
from vanedb import FlatIndex, Metric

# --- paste your embed() from step 2 here ---------------------------------

documents = [
    "VaneDB runs inside your application, with no database server.",
    "Cats sleep for most of the day.",
    "Embeddings turn text into lists of numbers.",
    "The train to Berlin leaves from platform 4.",
    "Vector search finds the nearest stored vectors to a query.",
]

vectors = embed(documents)

# The index dimension must match what the model returned. Read it, do not
# hardcode it.
index = FlatIndex(len(vectors[0]), Metric.COSINE)

# Ids are your own u64 handles. Here the id is the position in `documents`;
# in a real application it is your row id, and you keep the mapping yourself.
index.add_batch(range(len(documents)), vectors)

query = "how do I search text without running a server?"
[query_vector] = embed([query])

print(f"query: {query}\n")
for doc_id, distance in index.search(query_vector, 3):
    print(f"{distance:.3f}  {documents[doc_id]}")
```

Expected shape of the output — smaller distance means closer, and the first two
lines should be the two VaneDB sentences, not the cat:

```
query: how do I search text without running a server?

0.412  VaneDB runs inside your application, with no database server.
0.508  Vector search finds the nearest stored vectors to a query.
0.703  Embeddings turn text into lists of numbers.
```

The exact numbers depend on the model. The order is what matters.

## 4. Five rules that prevent most problems

1. **Same model for documents and queries.** Vectors from two different models
   are not comparable, and the search will silently return nonsense rather than
   fail.
2. **Dimension must match the model.** `FlatIndex(768, ...)` fed 1536 numbers
   raises an error. Take the dimension from `len(vectors[0])`.
3. **Use `Metric.COSINE` for text embeddings.** It is what text models are
   trained for. `L2` and `DOT` exist for other data.
4. **VaneDB stores only `(id, vector)`.** No text, no metadata, no filters. Keep
   your own id-to-document mapping — a dict, a JSON file, a SQLite table — and
   save it alongside the index.
5. **Changing the embedding model means re-embedding everything.** Old vectors
   in the index become garbage relative to new queries.

## 5. When the corpus grows

`FlatIndex` compares the query against every stored vector. That is exact and
fine for thousands of documents. Past roughly ten thousand, switch to
`ApproxIndex`, which searches a graph instead — much faster, and occasionally
misses a true neighbour. It is also the index that can be saved and reloaded, so
you embed once instead of on every start:

```python
import json
from pathlib import Path

from vanedb import ApproxIndex, Metric

index = ApproxIndex(len(vectors[0]), Metric.COSINE, capacity=len(documents))
index.add_batch(range(len(documents)), vectors)
index.save("corpus.vndb")
Path("corpus.json").write_text(json.dumps(documents))  # your id -> text map

# Later, in another process, with no calls to the embedding model:
index = ApproxIndex.load("corpus.vndb")
documents = json.loads(Path("corpus.json").read_text())
```

Raise `index.ef_search` (try 100) if the results look worse than `FlatIndex`
gave you; it buys recall with time. Embedding is usually much slower than
searching, so send documents to the provider in batches rather than one call per
document.

## 6. Other languages

The steps are identical: call the provider's HTTP API, get a list of floats,
hand it to VaneDB. See the [Rust quick start](../README.md#rust), the
[JavaScript guide](../vanedb-wasm/README.md) and the
[C guide](../vanedb-capi/README.md) for the storage half. Ollama's endpoint
(`POST http://localhost:11434/api/embed`) and OpenAI's
(`POST https://api.openai.com/v1/embeddings`) are plain JSON over HTTP from any
language.

## Checklist for an agent

Run in order; each step has an observable result.

1. `python -m pip install vanedb` (or `python -m pip install ./vanedb-py` from a
   checkout) → `import vanedb` succeeds.
2. Provider reachable:
   - Ollama: `ollama pull nomic-embed-text`, then the `curl` in step 2 returns
     JSON containing `embeddings`.
   - OpenAI: `OPENAI_API_KEY` is set in the environment; do not print it.
   - sentence-transformers: `python -c "import sentence_transformers"` succeeds.
3. `len(embed(["a", "b"])) == 2` and every row has the same length.
4. Build the index with `len(vectors[0])` as the dimension and `Metric.COSINE`.
5. Pass condition for step 3's script: the top hit for the VaneDB-flavoured
   query is one of the two VaneDB sentences. If it is the cat sentence, the
   query and the documents were embedded with different models, or `embed()`
   returned rows in a different order than it was given.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `pip install vanedb` finds no matching distribution | Only a prerelease is published so far. Add `--pre`, or install from a checkout. |
| `ConnectionRefusedError` on port 11434 | Ollama is not running. Start it: `ollama serve`. |
| Ollama returns `model ... not found` | `ollama pull nomic-embed-text` first. |
| `401` from OpenAI | `OPENAI_API_KEY` is unset, mistyped, or has no credit. |
| VaneDB raises a dimension error | The index dimension does not match the model's output length. Rebuild the index with `len(vectors[0])`. |
| First embedding call takes many seconds | The model is being loaded or downloaded. Subsequent calls are fast. |
| Results are unrelated to the query | Documents and query were embedded with different models, or the metric is not `COSINE`. |
| `KeyError: 'embeddings'` from Ollama | The older `/api/embeddings` endpoint returns `embedding` (singular) for a single `prompt`. Use `/api/embed` with `input` as shown. |
