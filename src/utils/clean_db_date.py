# src/utils/clean_db_date.py
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sqlalchemy import text

from src.utils.db import get_postgres_engine


def clean_database(cutoff_date: str = "2020-07-23 23:59:59"):
    engine = get_postgres_engine()
    with engine.connect() as conn:
        count_before = (
            conn.execute(text("SELECT COUNT(*) FROM silver.rawdata")).scalar() or 0
        )
        conn.execute(
            text("DELETE FROM silver.rawdata WHERE trans_date_trans_time > :cutoff"),
            {"cutoff": cutoff_date},
        )
        conn.commit()
        count_after = (
            conn.execute(text("SELECT COUNT(*) FROM silver.rawdata")).scalar() or 0
        )
        min_date, max_date = conn.execute(
            text(
                "SELECT MIN(trans_date_trans_time), MAX(trans_date_trans_time) FROM silver.rawdata"
            )
        ).fetchone()
        print(f" Nettoyage terminé : {count_before - count_after:,} lignes supprimées.")
        print(
            f" PostgreSQL silver.rawdata contient maintenant : {count_after:,} transactions du {min_date} au {max_date}."
        )


if __name__ == "__main__":
    cutoff = sys.argv[1] if len(sys.argv) > 1 else "2020-07-23 23:59:59"
    clean_database(cutoff)
