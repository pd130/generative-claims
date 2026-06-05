import ollama
import chromadb
from functools import lru_cache
from typing import Optional

CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "schema_fields"
EMBED_MODEL     = "mxbai-embed-large:335m"

# ---------------------------------------------------------------------------
# Module-level singletons — created once, reused across all calls
# ---------------------------------------------------------------------------
_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection]   = None


def get_collection(chroma_path: str = CHROMA_PATH) -> chromadb.Collection:
    """
    Return a module-level singleton ChromaDB collection.
    Avoids re-opening the client on every retrieval call.
    """
    global _client, _collection
    if _collection is None:
        _client     = chromadb.PersistentClient(path=chroma_path)
        _collection = _client.get_collection(COLLECTION_NAME)
    return _collection


@lru_cache(maxsize=256)
def embed_query(text: str) -> tuple[float, ...]:
    """
    Embed a query string, caching results so identical queries are not
    re-sent to Ollama.  Returns a tuple (hashable) for lru_cache compatibility;
    ChromaDB accepts lists, so callers convert with list().
    """
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return tuple(response["embeddings"][0])


def _build_query_text(partial_row: dict) -> str:
    """Stable, sorted query string so cache hits are order-independent."""
    context_str = ", ".join(f"{k}={v}" for k, v in sorted(partial_row.items()))
    return f"Insurance vehicle row constraints where {context_str}"


def retrieve_fields(
    partial_row: dict,
    n_results: int = 8,
    collection: Optional[chromadb.Collection] = None,
) -> list[dict]:
    """
    Retrieve the most relevant schema field constraints from ChromaDB.

    Optimisations vs original:
    - Singleton collection (no reconnect per call).
    - Cached query embeddings (same partial_row → no re-embed).
    - Sorted key order ensures cache hits despite dict insertion order.

    Args:
        partial_row : e.g. {"segment": "C2", "fuel_type": "Diesel"}
        n_results   : how many field docs to retrieve
        collection  : optionally pass a pre-opened collection

    Returns:
        List of dicts: {field, text, metadata, distance}
    """
    if collection is None:
        collection = get_collection()

    query_text   = _build_query_text(partial_row)
    query_vector = list(embed_query(query_text))   # convert tuple back to list

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "field":    field_id,
            "text":     results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": round(results["distances"][0][i], 4),
        }
        for i, field_id in enumerate(results["ids"][0])
    ]


def build_context_prompt(partial_row: dict, n_results: int = 8) -> str:
    """
    Retrieve relevant schema fields and format them as a
    ready-to-use prompt section for the Controller/Generator Agent.

    Result is deterministic for the same (partial_row, n_results) pair,
    so callers can cache it further if needed.
    """
    fields = retrieve_fields(partial_row, n_results=n_results)

    lines = [
        "=== Retrieved Schema Constraints ===",
        f"Context: generating row where {dict(sorted(partial_row.items()))}",
        "",
        *[f"• {f['text']}" for f in fields],
        "",
        "=== End of Constraints ===",
    ]
    return "\n".join(lines)


def reset_collection_cache() -> None:
    """Force re-open of the ChromaDB collection (e.g. after a reset/re-index)."""
    global _client, _collection
    _client     = None
    _collection = None
    embed_query.cache_clear()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    partial_row = {"segment": "C2", "fuel_type": "Diesel"}

    print("Partial row:", partial_row)
    print()

    results = retrieve_fields(partial_row, n_results=6)
    print("Top retrieved fields:")
    for r in results:
        print(f"  [{r['distance']:.4f}] {r['field']}")
        print(f"           {r['text'][:80]}...")
    print()

    # Second call – should hit embed cache
    print("Second call (cached embed)…")
    results2 = retrieve_fields(partial_row, n_results=6)
    print(f"  Got {len(results2)} results (embed was cached)\n")

    prompt_block = build_context_prompt(partial_row, n_results=6)
    print(prompt_block)