"""
Moves points (by id, read from pickle files) from "--limited --level 1 DENSE"
into other DENSE collections, deleting them from level 1 once the move is confirmed.

Usage:
    python move_points.py
"""
import logging
import os
from pathlib import Path
import pickle
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from qdrant_client.models import PointStruct, PointIdsList

from db.database import get_qdrant_client

LOG_PATH = Path(__file__).resolve().parent / "move_points.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

SOURCE_COLLECTION = "--limited --level 1 DENSE"

MOVES = [
    (
        "/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/hierarchical/indexed-at-30-06-26-limited-with-enriched/child/batch_1",
        "--limited --level 2 DENSE",
    ),
    (
        "/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/hierarchical/indexed-at-30-06-26-limited-with-enriched/enriched/batch_1",
        "--limited --level 3 DENSE",
    ),
]

RETRIEVE_BATCH = 100


def get_ids_from_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        chunk = pickle.load(f)
    ids = chunk["id"].tolist() if hasattr(chunk["id"], "tolist") else list(chunk["id"])
    return ids


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def move_ids(client, ids, dest_collection):
    """Fetch points by id from SOURCE_COLLECTION, upsert into dest_collection,
    then delete them from SOURCE_COLLECTION. Returns (moved_count, missing_ids)."""
    moved = 0
    missing = []

    for id_batch in chunked(ids, RETRIEVE_BATCH):
        records = client.retrieve(
            collection_name=SOURCE_COLLECTION,
            ids=id_batch,
            with_payload=True,
            with_vectors=True,
        )

        found_ids = {r.id for r in records}
        missing.extend([i for i in id_batch if i not in found_ids])

        if not records:
            continue

        points = [
            PointStruct(id=r.id, vector=r.vector, payload=r.payload)
            for r in records
        ]

        client.upsert(collection_name=dest_collection, wait=True, points=points)

        # Verify the upsert landed before deleting from source.
        verify = client.retrieve(
            collection_name=dest_collection,
            ids=[p.id for p in points],
            with_payload=False,
            with_vectors=False,
        )
        verified_ids = {r.id for r in verify}
        confirmed_ids = [p.id for p in points if p.id in verified_ids]
        unconfirmed_ids = [p.id for p in points if p.id not in verified_ids]
        if unconfirmed_ids:
            logger.warning(f"    {len(unconfirmed_ids)} points failed to verify in {dest_collection}, not deleting from source: {unconfirmed_ids}")

        if confirmed_ids:
            client.delete(
                collection_name=SOURCE_COLLECTION,
                points_selector=PointIdsList(points=confirmed_ids),
            )
            moved += len(confirmed_ids)

    return moved, missing


def main():
    client = get_qdrant_client()

    logger.info(f"=== Run start, logging to {LOG_PATH} ===")

    for path_, dest_collection in MOVES:
        logger.info(f"=== Moving points from {path_} into {dest_collection} ===")
        files = [f for f in os.listdir(path_) if f.endswith(".pkl")]

        for fname in files:
            pkl_path = os.path.join(path_, fname)
            ids = get_ids_from_pkl(pkl_path)
            logger.info(f"  {fname}: {len(ids)} ids")

            moved, missing = move_ids(client, ids, dest_collection)
            logger.info(f"    Moved {moved}/{len(ids)} points to {dest_collection}.")
            if missing:
                logger.info(f"    {len(missing)} ids not found in {SOURCE_COLLECTION}: {missing}")


if __name__ == "__main__":
    main()
