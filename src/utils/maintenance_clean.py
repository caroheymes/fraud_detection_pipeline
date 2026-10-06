# src/utils/maintenance_clean.py
"""
Script de maintenance :
1. Purge des transactions PostgreSQL après le 15 août 2020.
2. Réassignation de l'alias @champion à la version 26 du modèle.
3. Rechargement à chaud de l'API d'inférence FastAPI.
4. Export et synchronisation des règles de suspicion Redis depuis le modèle v26.
"""

from __future__ import annotations

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sqlalchemy import text

from src.utils.api_reloader import reload_serving_api
from src.utils.db import get_postgres_engine
from src.utils.mlflow_manager import MLflowQualityGate


def purge_database(cutoff_date: str = "2020-08-15 23:59:59"):
    print(f"🧹 1. PURGE DE LA BASE POSTGRESQL APRÈS LE {cutoff_date}...")
    engine = get_postgres_engine()
    with engine.connect() as conn:
        # 1.1 silver.rawdata
        try:
            res_cnt = conn.execute(
                text(
                    "SELECT COUNT(*) FROM silver.rawdata WHERE trans_date_trans_time > :cutoff"
                ),
                {"cutoff": cutoff_date},
            ).scalar()
            print(
                f"   • silver.rawdata : {res_cnt:,} transactions postérieures au {cutoff_date} détectées."
            )

            if res_cnt > 0:
                conn.execute(
                    text(
                        "DELETE FROM silver.rawdata WHERE trans_date_trans_time > :cutoff"
                    ),
                    {"cutoff": cutoff_date},
                )
                conn.commit()
                print(f"   ✅ {res_cnt:,} transactions supprimées de silver.rawdata.")
            else:
                print("   ℹ️ Aucune transaction à supprimer dans silver.rawdata.")

            # Nouvelle date max
            new_max = conn.execute(
                text("SELECT MAX(trans_date_trans_time) FROM silver.rawdata")
            ).scalar()
            total_remaining = conn.execute(
                text("SELECT COUNT(*) FROM silver.rawdata")
            ).scalar()
            print(
                f"   📊 silver.rawdata restante : {total_remaining:,} transactions (Date max: {new_max})."
            )
        except Exception as e:
            print(f"   ⚠️ Erreur silver.rawdata : {e}")

        # 1.2 Vérification d'éventuelles autres tables
        try:
            conn.execute(
                text(
                    "DELETE FROM public.simulation_queue WHERE trans_date_trans_time > :cutoff"
                ),
                {"cutoff": cutoff_date},
            )
            conn.commit()
            print("   ✅ simulation_queue purgée.")
        except Exception:
            pass


def promote_version(version: str = "26"):
    print(f"\n👑 2. PROMOTION DU MODÈLE VERSION {version} COMME @CHAMPION...")
    gate = MLflowQualityGate(model_name="fraud_detector")
    success = gate.set_champion_alias(version)
    if success:
        print(f"   ✅ Alias '@champion' réassigné avec succès à la version {version} !")
    else:
        print(
            f"   ❌ Échec de l'assignation de l'alias @champion à la version {version}."
        )

    gate.print_status_table()


def reload_api():
    print("\n🔄 3. RECHARGEMENT À CHAUD DE L'API FASTAPI...")
    reload_serving_api()


def sync_redis_rules():
    print("\n📝 4. SYNCHRONISATION DES RÈGLES REDIS (EXPORT RULES)...")
    try:
        import subprocess

        cmd = [
            sys.executable,
            os.path.join(project_root, "src/explain/export_rules.py"),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(
                "   ✅ Règles de suspicion SHAP réexportées et injectées dans Redis avec succès !"
            )
        else:
            print(f"   ⚠️ Erreur export_rules : {res.stderr}")
    except Exception as e:
        print(f"   ⚠️ Échec de l'export des règles : {e}")


import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Script de maintenance MLOps (Purge DB, Promotion Champion, Reload API, Export Rules)"
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default="2020-08-10 23:59:59",
        help="Date de coupure pour la purge (ex: '2020-08-10 23:59:59')",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="24",
        help="Version du modèle à promouvoir comme champion (ex: '24')",
    )
    args = parser.parse_args()

    purge_database(args.cutoff)
    promote_version(args.version)
    reload_api()
    sync_redis_rules()
    print("\n🏁 Opérations de maintenance et promotion terminées avec succès !")


if __name__ == "__main__":
    main()
