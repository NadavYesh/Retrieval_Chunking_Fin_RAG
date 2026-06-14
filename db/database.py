from qdrant_client import QdrantClient
import sqlite3
import os

# Qdrant configuration
QDRANT_URL = "http://localhost:6333"

def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, check_compatibility=True, cloud_inference=False)

# SQLite configuration
REGISTRY_DB_PATH = "/Users/nadavsmacbookair/Desktop/Thesis/code/data/chunks/sql_registry/registry.db"

def get_sqlite_conn():
    return sqlite3.connect(REGISTRY_DB_PATH)
