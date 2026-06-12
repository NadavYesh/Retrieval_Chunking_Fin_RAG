import sqlite3
from database import get_sqlite_conn, REGISTRY_DB_PATH

def init_registry_tables():
    query_table = '''CREATE TABLE IF NOT EXISTS POINT_COLLECTION_STATUS (
      point_uuid    TEXT NOT NULL,
      collection_id TEXT NOT NULL,
      indexed_at    DATETIME,
      payload_hash  TEXT,
      vector_hash   TEXT,
      status        TEXT DEFAULT 'pending',
      PRIMARY KEY (point_uuid, collection_id)
    )
    '''
    try:
        with get_sqlite_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query_table)
            conn.commit()
            print("Registry tables initialized.")
    except sqlite3.OperationalError as e:
        print("Error initializing registry tables:", e)

def list_collections():
    try:
        with get_sqlite_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM COLLECTIONS")
            result = cursor.fetchall()
            for row in result:
                print(row)
    except sqlite3.OperationalError as e:
        print("Error listing collections:", e)

if __name__ == "__main__":
    # init_registry_tables()
    # list_collections()
