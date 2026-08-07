# src/training/create_simulation_queue.py
# Exemples d'utilisation :
#   docker exec -t fraud-detection-ray-head python src/training/create_simulation_queue.py --duration-value 30 --duration-unit days
#   docker exec -t fraud-detection-ray-head python src/training/create_simulation_queue.py --duration-value 720 --duration-unit hours

import argparse
import glob
import os
import sys
from datetime import timedelta

import pandas as pd


def main():
    print("--- CRÉATION DE LA FILE D'ATTENTE DE SIMULATION DYNAMIQUE (30 ETAPES) ---")

    # 1. Parsing des arguments de ligne de commande
    parser = argparse.ArgumentParser(
        description="Génération de 30 fichiers de simulation pour l'inférence par lots."
    )
    parser.add_argument(
        "--duration-value",
        type=int,
        default=30,
        help="Valeur de la durée de la simulation (défaut: 30)",
    )
    parser.add_argument(
        "--duration-unit",
        type=str,
        choices=["days", "hours"],
        default="days",
        help="Unité de la durée: 'days' ou 'hours' (défaut: 'days')",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, "../../fraudTest.csv"))
    queue_dir = os.path.abspath(os.path.join(script_dir, "../../data/queue"))

    if not os.path.exists(csv_path):
        print(f"Erreur : Dataset {csv_path} introuvable.")
        sys.exit(1)

    # Création du dossier queue s'il n'existe pas
    os.makedirs(queue_dir, exist_ok=True)
    
    # Nettoyage préalable des anciens fichiers dans la file
    old_files = glob.glob(os.path.join(queue_dir, "*.csv"))
    for f in old_files:
        try:
            os.remove(f)
        except Exception:
            pass
    print(f"Nettoyé {len(old_files)} anciens fichiers de simulation dans : {queue_dir}")

    # 2. Chargement du dataset complet
    print("Chargement du dataset fraudTest.csv...")
    df = pd.read_csv(csv_path)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    # 3. Détermination dynamique de la date de début (date max dans PostgreSQL)
    db_user = os.getenv("POSTGRES_USER", "fraud-detection")
    db_password = os.getenv("POSTGRES_PASSWORD", "fraud-detection_password")
    db_host = os.getenv("POSTGRES_HOST", "postgres")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    db_db = os.getenv("POSTGRES_DB", "fraud-detection")
    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_db}"

    start_date = None
    try:
        import sqlalchemy
        engine = sqlalchemy.create_engine(db_url)
        with engine.connect() as conn:
            max_date_val = conn.execute(
                sqlalchemy.text("SELECT MAX(trans_date_trans_time) FROM silver.rawdata")
            ).scalar()
            if max_date_val:
                t = pd.to_datetime(max_date_val)
                if t.tzinfo is not None:
                    t = t.tz_localize(None)
                start_date = t
                print(f"[Simulation MLOps] Date max détectée dans Postgres : {start_date}")
    except Exception as e:
        print(f"[Simulation MLOps] Impossible de lire la date max Postgres, repli sur le défaut : {e}")

    if start_date is None:
        start_date = df["trans_date_trans_time"].min() + timedelta(days=30)
        print(f"[Simulation MLOps] Date de début par défaut : {start_date}")

    # Calcul de l'intervalle temporel pour découper exactement en 30 étapes
    if args.duration_unit == "days":
        total_delta = timedelta(days=args.duration_value)
    else:
        total_delta = timedelta(hours=args.duration_value)

    end_date = start_date + total_delta
    interval_delta = total_delta / 30.0

    print(f"Simulation du {start_date} au {end_date} ({args.duration_value} {args.duration_unit}).")
    print(f"Découpage en 30 fichiers de {interval_delta} chacun...\n")

    # 4. Génération et écriture des 30 fichiers
    for i in range(30):
        bin_start = start_date + i * interval_delta
        bin_end = bin_start + interval_delta

        # Filtrage
        df_bin = df[
            (df["trans_date_trans_time"] >= bin_start)
            & (df["trans_date_trans_time"] < bin_end)
        ].copy()

        # Construction du nom de fichier
        formatted_start = bin_start.strftime("%Y-%m-%d_%H-%M")
        file_name = f"step_{i+1:02d}_{formatted_start}.csv"
        file_path = os.path.join(queue_dir, file_name)

        df_bin.to_csv(file_path, index=False)
        print(f" -> [{i+1:02d}/30] {file_name} : {len(df_bin)} transactions écrites.")

    print("\n--- CRÉATION TERMINÉE AVEC SUCCÈS ---")
    print(f"La file d'attente contient 30 fichiers de simulation dans {queue_dir}.")


if __name__ == "__main__":
    main()
