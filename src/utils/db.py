# src/utils/db.py
"""
Connecteurs centralisés pour PostgreSQL et Redis.
Permet d'accéder aux bases de données de manière transparente dans les conteneurs Docker ou en local.
"""

from __future__ import annotations

import os

import redis
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def get_postgres_engine(custom_url: str | None = None, echo: bool = False) -> Engine:
    """
    Retourne une instance Engine SQLAlchemy configurée avec pool_pre_ping.
    """
    if custom_url:
        return create_engine(custom_url, pool_pre_ping=True, echo=echo)

    db_user = os.getenv("POSTGRES_USER", "fraud-detection")
    db_password = os.getenv("POSTGRES_PASSWORD", "fraud-detection_password")
    db_host = os.getenv("POSTGRES_HOST", "postgres")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    db_db = os.getenv("POSTGRES_DB", "fraud-detection")

    url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_db}"
    return create_engine(url, pool_pre_ping=True, echo=echo)


def get_redis_client(
    host: str | None = None, port: int = 6379, db: int = 0
) -> redis.Redis | None:
    """
    Retourne une instance de client Redis connectée, ou None en cas d'indisponibilité.
    """
    redis_host = host or os.getenv("REDIS_HOST", "redis")
    try:
        r = redis.Redis(host=redis_host, port=port, db=db, decode_responses=True)
        r.ping()
        return r
    except Exception:
        # Essai de repli sur localhost
        try:
            r = redis.Redis(host="localhost", port=port, db=db, decode_responses=True)
            r.ping()
            return r
        except Exception:
            return None
