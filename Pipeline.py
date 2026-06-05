"""
Pipeline.py  –  End-to-end synthetic row generation pipeline

Stages
------
1. Profiling   : read CSV → build schema.json          (schemagenerator)
2. Embedding   : schema.json → embeddings.json         (vector.py logic, parallel)
3. Indexing    : embeddings.json → ChromaDB            (indexer.py logic)
4. Generation  : ChromaDB + Agents → synthetic rows    (agents.py logic, parallel workers)
5. Export      : list[dict] → output CSV

Performance improvements
------------------------
- Stage 2: fields are embedded concurrently (ThreadPoolExecutor).
- Stage 3: ChromaDB upsert is batched in one call instead of per-field.
- Stage 4: rows are generated in parallel worker threads; each worker holds
           its own ChromaDB collection handle so there's no contention on a
           shared connection.
- Retriever: singleton collection + LRU-cached query embeddings (see retriever.py).
"""
import random
import os
import json
import time
import threading
import pandas as pd
import ollama
import chromadb
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

from Profiler.schemagenerator import build_schema, stats
from retriever import build_context_prompt, get_collection, reset_collection_cache
from agents import (
    build_generator_prompt,
    call_ollama,
)
# ---------------------------------------------------------------------------
# Config  (edit here or override via env vars)
# ---------------------------------------------------------------------------
BASE_DIR        = os.getenv("PIPELINE_BASE_DIR", r"D:\Virtual Environment\project\PBL1")
CSV_PATH        = os.getenv("PIPELINE_CSV",      "data.csv")
SCHEMA_PATH     = os.getenv("PIPELINE_SCHEMA",   "schema.json")
EMBEDDINGS_PATH = os.getenv("PIPELINE_EMBED",    "embeddings.json")
CHROMA_PATH     = os.getenv("PIPELINE_CHROMA",   "./chroma_db")
OUTPUT_PATH     = os.getenv("PIPELINE_OUTPUT",   "synthetic_data.csv")

EMBED_MODEL     = "mxbai-embed-large:335m"
COLLECTION_NAME = "schema_fields"

# Generation settings
ROWS_TO_GENERATE    = int(os.getenv("ROWS_TO_GENERATE",    "1500"))
MAX_RETRIES         = int(os.getenv("MAX_RETRIES",         "3"))
EMBED_WORKERS       = int(os.getenv("EMBED_WORKERS",       "8"))   # parallel embed threads
GENERATION_WORKERS  = int(os.getenv("GENERATION_WORKERS",  "1"))   # safe for RTX 4070 laptop; try 3 if stable
CHROMA_BATCH_SIZE   = int(os.getenv("CHROMA_BATCH_SIZE",   "100")) # upsert batch size
_ollama_sem = threading.Semaphore(1)

# ===========================================================================
# Stage 1 – Profiling
# ===========================================================================

def stage_profiling(csv_path: str, schema_path: str) -> tuple[pd.DataFrame, dict]:

    df = pd.read_csv(csv_path)
    print(f"  Loaded '{csv_path}'  →  {df.shape[0]} rows × {df.shape[1]} cols")

    schema = build_schema(df)
    schema = stats(df, schema)

    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=4)

    print(f"  Schema saved to '{schema_path}'  ({len(schema)} fields)")
    return df, schema


# ===========================================================================
# Stage 2 – Embedding  (parallel)
# ===========================================================================

def _field_to_text(field: str, rules: dict) -> str:
    t = rules["type"]
    if rules.get("values") == [0, 1]:
        pct = rules.get("mean", 0) * 100
        return (
            f"Field '{field}': binary feature (0=absent, 1=present). "
            f"{pct:.1f}% of vehicles have this feature."
        )
    elif t == "categorical":
        vals    = ", ".join(str(v) for v in rules["values"])
        top3    = sorted(rules.get("value_counts", {}).items(), key=lambda x: -x[1])[:3]
        top3_str = ", ".join(f"{k} ({v} rows)" for k, v in top3)
        return (
            f"Field '{field}': categorical with {rules['n_unique']} values: [{vals}]. "
            f"Most common: {top3_str}."
        )
    else:
        return (
            f"Field '{field}': {t}, valid range [{rules['min']}, {rules['max']}], "
            f"mean={rules['mean']}, std={rules['std']}, "
            f"p25={rules['p25']}, median={rules['p50']}, p75={rules['p75']}. "
            f"Nullable: {rules.get('nullable', False)}."
        )


def _embed_one(field: str, rules: dict) -> tuple[str, dict]:
    text     = _field_to_text(field, rules)
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return field, {
        "text":      text,
        "embedding": response["embeddings"][0],
        "metadata": {
            "type":      rules["type"],
            "is_binary": rules.get("values") == [0, 1],
        },
    }


def stage_embedding(schema: dict, embeddings_path: str,
                    max_workers: int = EMBED_WORKERS) -> dict:
    print("\n" + "=" * 60)
    print(f"STAGE 2 – Embedding  (parallel, {max_workers} workers)")
    print("=" * 60)

    embeddings_store: dict = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[Future, str] = {
            executor.submit(_embed_one, field, rules): field
            for field, rules in schema.items()
        }
        for future in as_completed(futures):
            field, entry = future.result()
            embeddings_store[field] = entry
            print(f"  Embedded: {field}  ({len(entry['embedding'])} dims)")

    with open(embeddings_path, "w") as f:
        json.dump(embeddings_store, f, indent=4)

    return embeddings_store


# ===========================================================================
# Stage 3 – Indexing  (batched upsert)
# ===========================================================================

def stage_indexing(
    embeddings_path: str,
    chroma_path: str,
    reset: bool = False,
    batch_size: int = CHROMA_BATCH_SIZE,
) -> chromadb.Collection:

    with open(embeddings_path) as f:
        store = json.load(f)

    print(f"  Loaded {len(store)} embeddings from '{embeddings_path}'")

    client = chromadb.PersistentClient(path=chroma_path)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"  Dropped existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass
        reset_collection_cache()   # clear the singleton in retriever.py

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set(collection.get()["ids"])
    new_items    = [(k, v) for k, v in store.items() if k not in existing_ids]

    if not new_items:
        print("  All fields already indexed. Nothing to add.")
    else:
        # Batch upsert instead of one-by-one
        for start in range(0, len(new_items), batch_size):
            batch = new_items[start : start + batch_size]
            collection.add(
                ids        = [k          for k, v in batch],
                embeddings = [v["embedding"] for k, v in batch],
                documents  = [v["text"]      for k, v in batch],
                metadatas  = [v["metadata"]  for k, v in batch],
            )
        print(f"  Indexed {len(new_items)} new fields  →  '{COLLECTION_NAME}'")
    return collection


# ===========================================================================
# Stage 4 – Generation  (parallel row workers)
# ===========================================================================

def _parse_json_response(raw: str) -> dict:
    import re

    # ------------------------------------------------------------------ #
    # Phase 1 – Sanitise: remove all wrapper noise from the raw string    #
    # ------------------------------------------------------------------ #
    text = raw.strip()

    # Remove <think>...</think> blocks (deepseek-r1, Qwen3, etc.)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Remove markdown code fences  (```json ... ``` or ``` ... ```)
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$",          "", text)
    text = text.strip()

    # ------------------------------------------------------------------ #
    # Phase 2 – Extract: try progressively looser parse strategies        #
    # ------------------------------------------------------------------ #

    # Strategy 1 – direct parse (ideal: model returned clean JSON)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2 – slice from first '{' to last '}' (leading/trailing prose)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 3 – greedy regex for {...} or [...] (whitespace-heavy responses)
    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    raise json.JSONDecodeError("No valid JSON found in response", text, 0)


def _python_validate(full_row: dict, schema: dict) -> tuple[bool, str]:
    """Pure-Python validator — no LLM call. Checks types, ranges, categoricals."""
    violations = []
    for field, rules in schema.items():
        if field not in full_row:
            violations.append(f"'{field}' is missing")
            continue
        val = full_row[field]
        t   = rules.get("type")
        if rules.get("values") == [0, 1]:
            if val not in (0, 1):
                violations.append(f"'{field}' must be 0 or 1, got {val!r}")
        elif t == "categorical":
            allowed = rules.get("values", [])
            if val not in allowed:
                violations.append(f"'{field}' value {val!r} not in allowed list")
        elif t in ("int", "float", "integer", "numeric"):
            try:
                fval = float(val)
            except (TypeError, ValueError):
                violations.append(f"'{field}' is not numeric: {val!r}")
                continue
            lo, hi = rules.get("min"), rules.get("max")
            if lo is not None and fval < lo:
                violations.append(f"'{field}' = {fval} below min {lo}")
            if hi is not None and fval > hi:
                violations.append(f"'{field}' = {fval} above max {hi}")
    if violations:
        return False, "; ".join(violations[:5])
    return True, ""

def _clamp_to_schema(generated: dict, schema: dict) -> dict:
    """
    Hard-clamp numeric fields to their schema min/max.
    Prevents the LLM from overshooting on fields like vehicle_age.
    """
    clamped = dict(generated)
    for field, rules in schema.items():
        if field not in clamped:
            continue
        t = rules.get("type")
        if t not in ("int", "float", "integer", "numeric"):
            continue
        lo, hi = rules.get("min"), rules.get("max")
        try:
            val = float(clamped[field])
            if lo is not None:
                val = max(val, lo)
            if hi is not None:
                val = min(val, hi)
            # preserve int type if schema says int
            clamped[field] = int(val) if t in ("int", "integer") else val
        except (TypeError, ValueError):
            pass
    return clamped

def generate_one_row(
    partial_row: dict,
    collection: chromadb.Collection,
    schema: dict,
    max_retries: int = MAX_RETRIES,
) -> dict | None:
    """Generator → Python-Validator loop. One LLM call per attempt."""
    feedback = ""

    for attempt in range(1, max_retries + 1):
        gen_prompt   = build_generator_prompt(partial_row, feedback=feedback, schema=schema)

        with _ollama_sem:
            gen_response = call_ollama(
                prompt=gen_prompt,
                system="Return ONLY a valid JSON object with the required fields. No explanation.",
                temperature=0.8,
            )

        try:
            generated_row = _parse_json_response(str(gen_response))
        except json.JSONDecodeError as e:
            feedback = f"Your response was not valid JSON: {e}. Return ONLY a raw JSON object."
            print(f"    [attempt {attempt}] JSON parse error – retrying…")
            continue

        # ── clamp numerics to schema bounds before validation ──
        generated_row = _clamp_to_schema(generated_row, schema)
        # ───────────────────────────────────────────────────────

        full_row = {**partial_row, **generated_row}

        passed, feedback = _python_validate(full_row, schema)
        if passed:
            print(f"    [attempt {attempt}]  Row validated successfully")
            return full_row
        else:
            print(f"    [attempt {attempt}]  Violations: {feedback}")

    print(f"     Exceeded {max_retries} retries – skipping row")
    return None


def _get_categoricals(schema: dict, fields: list[str]) -> dict:
    return {
        f: schema[f]["values"]
        for f in fields
        if f in schema and schema[f].get("type") == "categorical"
    }


_thread_local = threading.local()

def _get_thread_collection(chroma_path: str) -> chromadb.Collection:
    """One ChromaDB client per worker thread, opened once and reused."""
    if not hasattr(_thread_local, "collection"):
        client = chromadb.PersistentClient(path=chroma_path)
        _thread_local.collection = client.get_collection(COLLECTION_NAME)
    return _thread_local.collection


def _worker(
    row_index: int,
    n_rows: int,
    schema: dict,
    seed_fields: dict,
    chroma_path: str,
) -> dict | None:
    collection  = _get_thread_collection(chroma_path)
    partial_row = dict(seed_fields)
    print(f"\n  Row {row_index}/{n_rows}")

    anchor_fields  = ["segment", "fuel_type", "region_code"]
    allowed_values = _get_categoricals(schema, anchor_fields)
    # Only randomly anchor fields that were NOT already fixed by seed_fields.
    # Previously this unconditionally overwrote seed values — scenario seeds
    # like {"fuel_type": "Diesel"} were silently replaced by a random pick.
    anchors = {
        k: random.choice(v)
        for k, v in allowed_values.items()
        if v and k not in seed_fields          # ← don't clobber seeded fields
    }
    partial_row.update(anchors)
    print(f"  Anchors: {partial_row}")

    return generate_one_row(
        partial_row=partial_row,
        collection=collection,
        schema=schema,
        max_retries=MAX_RETRIES,
    )


CHECKPOINT_PATH  = os.getenv("PIPELINE_CHECKPOINT", "checkpoint.jsonl")
_checkpoint_lock = threading.Lock()


def _append_checkpoint(row: dict, path: str) -> None:
    with _checkpoint_lock:
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")


def _load_checkpoint(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def stage_generation(
    schema: dict,
    collection: chromadb.Collection,
    n_rows: int = ROWS_TO_GENERATE,
    seed_fields: dict | None = None,
    chroma_path: str = CHROMA_PATH,
    max_workers: int = GENERATION_WORKERS,
    checkpoint_path: str = CHECKPOINT_PATH,
    resume: bool = True,
) -> list[dict]:
    print("Generating")

    seed_fields    = seed_fields or {}
    completed_rows = _load_checkpoint(checkpoint_path) if resume else []
    already_done   = len(completed_rows)

    if already_done:
        print(f"  Resuming: {already_done} rows already completed from checkpoint.")

    remaining = n_rows - already_done
    if remaining <= 0:
        print("  All rows already generated.")
        return completed_rows

    rows = list(completed_rows)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _worker, already_done + i, n_rows, schema, seed_fields, chroma_path
            ): already_done + i
            for i in range(1, remaining + 1)
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                rows.append(result)
                _append_checkpoint(result, checkpoint_path)

    print(f"\n  Generated {len(rows)} valid rows out of {n_rows} requested")
    return rows


# ===========================================================================
# Stage 5 – Export
# ===========================================================================

def stage_export(rows: list[dict], output_path: str) -> pd.DataFrame:

    if not rows:
        print("  No rows to export.")
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False)
    print(f"  {len(df_out)} rows  ×  {len(df_out.columns)} cols  →  '{output_path}'")
    return df_out


# ===========================================================================
# Main entry point
# ===========================================================================

def run_pipeline(
    csv_path:           str  = CSV_PATH,
    schema_path:        str  = SCHEMA_PATH,
    embeddings_path:    str  = EMBEDDINGS_PATH,
    chroma_path:        str  = CHROMA_PATH,
    output_path:        str  = OUTPUT_PATH,
    n_rows:             int  = ROWS_TO_GENERATE,
    reset_index:        bool = False,
    seed_fields:        dict | None = None,
    skip_profiling:     bool = False,
    skip_embedding:     bool = False,
    skip_indexing:      bool = False,
    embed_workers:      int  = EMBED_WORKERS,
    generation_workers: int  = GENERATION_WORKERS,
    resume:             bool = True,
    checkpoint_path:    str  = CHECKPOINT_PATH,
) -> pd.DataFrame:
    """
    Run the full pipeline end-to-end.

    Skip flags let you resume from a checkpoint:
      skip_profiling  – reuse existing schema.json
      skip_embedding  – reuse existing embeddings.json
      skip_indexing   – reuse existing ChromaDB

    Tuning knobs:
      embed_workers      – threads for Stage 2 (parallel Ollama embed calls)
      generation_workers – threads for Stage 4 (parallel row generation)
    """
    print("Starting pipeline")

    # Stage 1
    if skip_profiling:
        print("\n[SKIP] Profiling – loading existing schema.json")
        with open(schema_path) as f:
            schema = json.load(f)
    else:
        _, schema = stage_profiling(csv_path, schema_path)

    # Stage 2
    if skip_embedding:
        print("\n[SKIP] Embedding – using existing embeddings.json")
    else:
        stage_embedding(schema, embeddings_path, max_workers=embed_workers)

    # Stage 3
    if skip_indexing:
        print("\n[SKIP] Indexing – connecting to existing ChromaDB")
        collection = get_collection(chroma_path)
    else:
        collection = stage_indexing(embeddings_path, chroma_path, reset=reset_index)

    # Stage 4
    rows = stage_generation(
        schema=schema,
        collection=collection,
        n_rows=n_rows,
        seed_fields=seed_fields,
        chroma_path=chroma_path,
        max_workers=generation_workers,
        checkpoint_path=checkpoint_path,
        resume=resume,
    )

    # Stage 5
    df_out = stage_export(rows, output_path)

    print("  PIPELINE COMPLETE")
    return df_out


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.chdir(BASE_DIR)

    run_pipeline(
        csv_path           = CSV_PATH,
        schema_path        = SCHEMA_PATH,
        embeddings_path    = EMBEDDINGS_PATH,
        chroma_path        = CHROMA_PATH,
        output_path        = OUTPUT_PATH,
        n_rows             = ROWS_TO_GENERATE,
        reset_index        = False,
        seed_fields        = None,
        skip_profiling     = False,
        skip_embedding     = False,
        skip_indexing      = False,
        embed_workers      = EMBED_WORKERS,       # tune: 4–16 depending on Ollama concurrency
        generation_workers = GENERATION_WORKERS,  # tune: 2–8 depending on RAM / GPU
    )