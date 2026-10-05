# src/training/create_simulation_queue.py
# Exemples d'utilisation :
#   docker exec -t fraud-detection-ray-head python src/training/create_simulation_queue.py --duration-value 30 --duration-unit days
#   docker exec -t fraud-detection-ray-head python src/training/create_simulation_queue.py --duration-value 720 --duration-unit hours

import argparse
import os
import sys
from datetime import timedelta

import pandas as pd
from sqlalchemy import text

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.db import get_postgres_engine


def main():
    # 1. Parsing des arguments de ligne de commande
    parser = argparse.ArgumentParser(
        description="Génération de fichiers de simulation pour l'inférence par lots."
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
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Nombre d'étapes/fichiers à générer (défaut: 30)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Date de début forcée (format: 'YYYY-MM-DD HH:MM:SS'). Par défaut: MAX(trans_date_trans_time) dans PostgreSQL.",
    )
    args = parser.parse_args()

    print(
        f"--- CRÉATION DE LA FILE D'ATTENTE DE SIMULATION DYNAMIQUE ({args.steps} ETAPES) ---"
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, "../../fraudTest.csv"))
    queue_dir = os.path.abspath(os.path.join(script_dir, "../../data/queue"))

    if not os.path.exists(csv_path):
        print(f"Erreur : Dataset {csv_path} introuvable.")
        sys.exit(1)

    # Création du dossier queue s'il n'existe pas
    os.makedirs(queue_dir, exist_ok=True)

    # 2. Chargement du dataset complet
    print("Chargement du dataset fraudTest.csv...")
    df = pd.read_csv(csv_path)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    # 3. Détermination dynamique de la date de début (date max dans PostgreSQL)
    start_date = None

    if args.start_date:
        try:
            start_date = pd.to_datetime(args.start_date)
            print(
                f"[Simulation MLOps] Date de début spécifiée par argument CLI : {start_date}"
            )
        except Exception as e:
            print(
                f"[Simulation MLOps] Format de date invalide pour --start-date ({args.start_date}) : {e}"
            )

    if start_date is None:
        try:
            engine = get_postgres_engine()
            with engine.connect() as conn:
                max_date_val = conn.execute(
                    text("SELECT MAX(trans_date_trans_time) FROM silver.rawdata")
                ).scalar()
                if max_date_val:
                    t = pd.to_datetime(max_date_val)
                    if t.tzinfo is not None:
                        t = t.tz_localize(None)
                    start_date = t
                    print(
                        f"[Simulation MLOps] ✅ Date MAX détectée dans PostgreSQL : {start_date}"
                    )
        except Exception as e:
            print(f"[Simulation MLOps] Connexion PostgreSQL non disponible ({e}).")

    if start_date is None:
        start_date = df["trans_date_trans_time"].min() + timedelta(days=30)
        print(f"[Simulation MLOps] ⚠️ Repli sur la date par défaut : {start_date}")

    # 3.5 Vérifier si des fichiers sont déjà en attente dans la queue (pour accumuler sans écraser)
    max_queue_date = None
    try:
        with os.scandir(queue_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".csv"):
                    # Nom standard: step_XX_YYYY-MM-DD_HH-MM.csv
                    parts = entry.name.replace(".csv", "").split("_")
                    if len(parts) >= 4:
                        dt_str = f"{parts[2]} {parts[3].replace('-', ':')}:00"
                        try:
                            file_dt = pd.to_datetime(dt_str)
                            if max_queue_date is None or file_dt > max_queue_date:
                                max_queue_date = file_dt
                        except Exception:
                            pass
    except Exception as q_err:
        print(f"[Simulation MLOps] Erreur lecture queue : {q_err}")

    if max_queue_date is not None:
        if max_queue_date > start_date:
            print(
                f"[Simulation MLOps] 📁 Fichiers en attente dans la queue : décalage du début au {max_queue_date} pour accumuler."
            )
            start_date = max_queue_date

    # Calcul de l'intervalle temporel pour découper exactement en args.steps étapes
    if args.duration_unit == "days":
        total_delta = timedelta(days=args.duration_value)
    else:
        total_delta = timedelta(hours=args.duration_value)

    end_date = start_date + total_delta
    interval_delta = total_delta / float(args.steps)

    print(
        f"Simulation du {start_date} au {end_date} ({args.duration_value} {args.duration_unit})."
    )
    print(f"Découpage en {args.steps} fichiers de {interval_delta} chacun...\n")

    # 4. Génération et écriture des args.steps fichiers
    for i in range(args.steps):
        bin_start = start_date + i * interval_delta
        bin_end = bin_start + interval_delta

        # Filtrage sans doublons avec l'instant initial
        if i == 0:
            df_bin = df[
                (df["trans_date_trans_time"] > bin_start)
                & (df["trans_date_trans_time"] <= bin_end)
            ].copy()
        else:
            df_bin = df[
                (df["trans_date_trans_time"] > bin_start)
                & (df["trans_date_trans_time"] <= bin_end)
            ].copy()

        # Construction du nom de fichier
        formatted_start = bin_start.strftime("%Y-%m-%d_%H-%M")
        file_name = f"step_{i + 1:02d}_{formatted_start}.csv"
        file_path = os.path.join(queue_dir, file_name)

        if df_bin.empty:
            print(
                f" -> [{i + 1:02d}/{args.steps}] {file_name} : Ignoré (aucune transaction sur cet intervalle)."
            )
            continue

        df_bin.to_csv(file_path, index=False)
        print(
            f" -> [{i + 1:02d}/{args.steps}] {file_name} : {len(df_bin)} transactions écrites."
        )

    print("\n--- CRÉATION TERMINÉE AVEC SUCCÈS ---")
    print(
        f"La file d'attente contient {args.steps} fichiers de simulation dans {queue_dir}."
    )


if __name__ == "__main__":
    main()
