import json
import chromadb

EMBEDDINGS_PATH = "embeddings.json"
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "schema_fields"


def build_index(embeddings_path: str = EMBEDDINGS_PATH,
                chroma_path: str   = CHROMA_PATH,
                reset: bool        = False) -> chromadb.Collection:
    """
    Load embeddings.json into a ChromaDB persistent collection.

    Args:
        embeddings_path : path to the embeddings.json produced by vector.py
        chroma_path     : directory where ChromaDB will persist data
        reset           : if True, drop and recreate the collection

    Returns:
        The ChromaDB collection object.
    """
    # --- Load embeddings ---
    with open(embeddings_path) as f:
        store = json.load(f)

    print(f"Loaded {len(store)} field embeddings from '{embeddings_path}'")

    # --- Connect to ChromaDB ---
    client = chromadb.PersistentClient(path=chroma_path)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Dropped existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}   # cosine similarity for text embeddings
    )

    # Skip fields already indexed (safe to re-run)
    existing_ids = set(collection.get()["ids"])
    new_fields   = {k: v for k, v in store.items() if k not in existing_ids}

    if not new_fields:
        print("All fields already indexed. Nothing to add.")
        return collection

    # --- Add to ChromaDB ---
    collection.add(
        ids        = list(new_fields.keys()),
        embeddings = [v["embedding"] for v in new_fields.values()],
        documents  = [v["text"]      for v in new_fields.values()],
        metadatas  = [v["metadata"]  for v in new_fields.values()],
    )

    print(f"Indexed {len(new_fields)} fields into collection '{COLLECTION_NAME}'")
    print(f"ChromaDB persisted at: {chroma_path}")
    return collection


if __name__ == "__main__":
    collection = build_index()

    # Quick sanity check
    total = collection.count()
    print(f"\nCollection '{COLLECTION_NAME}' now has {total} documents.")