import ollama
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

EMBED_MODEL = "mxbai-embed-large:335m"
MAX_WORKERS = 8  # tune to your Ollama server's concurrency limit


def field_to_text(field: str, rules: dict) -> str:
    """Convert a single schema field to a natural language string for embedding."""
    t = rules["type"]

    if rules.get("values") == [0, 1]:
        pct = rules.get("mean", 0) * 100
        return (
            f"Field '{field}': binary feature (0=absent, 1=present). "
            f"{pct:.1f}% of vehicles have this feature."
        )
    elif t == "categorical":
        vals = ", ".join(str(v) for v in rules["values"])
        top3 = sorted(rules.get("value_counts", {}).items(), key=lambda x: -x[1])[:3]
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


def _embed_field(field: str, rules: dict) -> tuple[str, dict]:
    """Embed a single field and return (field_name, store_entry)."""
    text = field_to_text(field, rules)
    response = ollama.embed(model=EMBED_MODEL, input=text)
    entry = {
        "text": text,
        "embedding": response["embeddings"][0],
        "metadata": {
            "type": rules["type"],
            "is_binary": rules.get("values") == [0, 1],
        },
    }
    return field, entry


def build_embeddings(schema: dict, max_workers: int = MAX_WORKERS) -> dict:
    """
    Embed all schema fields concurrently.

    Returns:
        embeddings_store dict keyed by field name.
    """
    embeddings_store = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_embed_field, field, rules): field
            for field, rules in schema.items()
        }
        for future in as_completed(futures):
            field, entry = future.result()
            embeddings_store[field] = entry
            print(f"  Embedded: {field} ({len(entry['embedding'])} dims)")

    return embeddings_store


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    with open("schema.json") as f:
        schema = json.load(f)

    embeddings_store = build_embeddings(schema)

    with open("embeddings.json", "w") as f:
        json.dump(embeddings_store, f, indent=4)

    print(f"\nDone. {len(embeddings_store)} field embeddings saved to embeddings.json")