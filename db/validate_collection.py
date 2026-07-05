"""
Validates Qdrant collections:
  1. For each (pkl_dir, collection_name) in PKL_DIR_CHECKS, checks that all ids
     found in the pickle files under pkl_dir are present in that collection.
  2. Checks that every point in CHILD_COLLECTION has a payload "parent_id" that
     matches an existing point id in PARENT_COLLECTION.

Usage:
    python -m db.validate_collection
"""
import os
from pathlib import Path
import pickle
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db.database import get_qdrant_client

# --- Global parameters -------------------------------------------------

PARENT_COLLECTION = "--limited --level 1 DENSE"
CHILD_COLLECTION = "--limited --level 2 DENSE"

# Each entry: (directory of .pkl files, collection to check ids against)
PKL_DIR_CHECKS = [
    (
        "/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/hierarchical/indexed-at-30-06-26-limited-with-enriched/header/batch_1",
        PARENT_COLLECTION,
    ),
    (
        "/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/hierarchical/indexed-at-30-06-26-limited-with-enriched/enriched/batch_1",
        PARENT_COLLECTION,
    ),
]

RETRIEVE_BATCH = 100
SCROLL_BATCH = 250


# --- Shared helpers ------------------------------------------------------

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def ids_exist(client, collection_name, ids):
    """Returns the subset of `ids` that exist in `collection_name`."""
    existing = set()
    for id_batch in chunked(ids, RETRIEVE_BATCH):
        records = client.retrieve(
            collection_name=collection_name,
            ids=id_batch,
            with_payload=False,
            with_vectors=False,
        )
        existing.update(r.id for r in records)
    return existing


def iter_all_points(client, collection_name, batch_size=SCROLL_BATCH):
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for r in records:
            yield r
        if offset is None:
            break


# --- Check 1: pkl ids present in a collection ----------------------------

def get_ids_from_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        chunk = pickle.load(f)
    return chunk["id"].tolist() if hasattr(chunk["id"], "tolist") else list(chunk["id"])


def check_pkl_ids_present(client, pkl_dir, collection_name):
    print(f"\n=== Checking ids from {pkl_dir} are present in {collection_name} ===")
    files = [f for f in os.listdir(pkl_dir) if f.endswith(".pkl")]

    total_ids = 0
    total_missing = 0

    for fname in files:
        pkl_path = os.path.join(pkl_dir, fname)
        ids = get_ids_from_pkl(pkl_path)
        found = ids_exist(client, collection_name, ids)
        missing = [i for i in ids if i not in found]

        total_ids += len(ids)
        total_missing += len(missing)

        status = "OK" if not missing else "MISSING"
        print(f"[{status}] {fname}: {len(ids) - len(missing)}/{len(ids)} present in {collection_name}")
        if missing:
            print(f"    missing ids: {missing}")

    print(f"Total: {total_ids - total_missing}/{total_ids} ids present in {collection_name}")
    if total_missing == 0:
        print("All ids are present.")
    else:
        print(f"{total_missing} ids are missing.")


# --- Check 2: child parent_id resolves to an existing parent point -------

def check_parent_ids(client, child_collection, parent_collection):
    print(f"\n=== Checking {child_collection} points have a valid parent_id in {parent_collection} ===")

    no_parent_id = []
    parent_id_by_child = {}

    for point in iter_all_points(client, child_collection):
        parent_id = (point.payload or {}).get("parent_id")
        if parent_id is None:
            no_parent_id.append(point.id)
        else:
            parent_id_by_child[point.id] = parent_id

    total_children = len(no_parent_id) + len(parent_id_by_child)
    print(f"Scanned {total_children} points in {child_collection}.")

    if no_parent_id:
        print(f"{len(no_parent_id)} points have no parent_id in payload: {no_parent_id}")

    unique_parent_ids = list(set(parent_id_by_child.values()))
    existing_parent_ids = ids_exist(client, parent_collection, unique_parent_ids)
    missing_parent_ids = set(unique_parent_ids) - existing_parent_ids

    orphans = {
        child_id: parent_id
        for child_id, parent_id in parent_id_by_child.items()
        if parent_id in missing_parent_ids
    }

    print(f"{len(unique_parent_ids)} unique parent_ids referenced; "
          f"{len(existing_parent_ids)} found in {parent_collection}, "
          f"{len(missing_parent_ids)} missing.")

    if orphans:
        print(f"{len(orphans)} child points reference a missing parent_id:")
        for child_id, parent_id in orphans.items():
            print(f"    child {child_id} -> missing parent_id {parent_id}")
    else:
        print("All child points with a parent_id resolve to an existing parent.")

    if not no_parent_id and not orphans:
        print("All points in the child collection are fully valid.")


def main():
    client = get_qdrant_client()

    for pkl_dir, collection_name in PKL_DIR_CHECKS:
        check_pkl_ids_present(client, pkl_dir, collection_name)

    check_parent_ids(client, CHILD_COLLECTION, PARENT_COLLECTION)


if __name__ == "__main__":
    main()
