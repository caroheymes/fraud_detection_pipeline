# src/training/create_simulation_queue.py
# docker exec -t fraud-detection-ray-head python src/training/create_simulation_queue.py

import os
import sys
from datetime import timedelta

import pandas as pd


def main():
    print("--- CRÉATION DE LA FILE D'ATTENTE DE SIMULATION (BINS DE 1 MINUTE) ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, "../../fraudTest.csv"))
    queue_dir = os.path.abspath(os.path.join(script_dir, "../../data/queue"))

    if not os.path.exists(csv_path):
        print(f"Erreur : Dataset {csv_path} introuvable.")
        sys.exit(1)

    # Création du dossier queue s'il n'existe pas
    os.makedirs(queue_dir, exist_ok=True)
    print(f"Les fichiers de simulation seront écrits dans : {queue_dir}")

    # 1. Chargement du dataset complet
    print("Chargement du dataset fraudTest.csv...")
    df = pd.read_csv(csv_path)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    # 2. Filtrage de la fenêtre de simulation (7 jours après l'entraînement initial)
    start_date = df["trans_date_trans_time"].min() + timedelta(days=30)
    end_date = start_date + timedelta(days=7)

    print(f"Extraction des transactions du {start_date} au {end_date}...")
    df_sim = df[
        (df["trans_date_trans_time"] >= start_date)
        & (df["trans_date_trans_time"] < end_date)
    ].copy()

    # 3. Création de la clé temporelle au format YYYY-MM-DD_HH-MM
    df_sim["time_key"] = df_sim["trans_date_trans_time"].dt.strftime("%Y-%m-%d_%H-%M")

    # 4. Groupement et écriture des mini-CSV
    groups = df_sim.groupby("time_key")
    total_groups = len(groups)
    print(f"Total de fichiers de 1 minute à générer : {total_groups}")

    count = 0
    for time_key, group in groups:
        # On supprime la colonne temporaire time_key avant d'écrire
        clean_group = group.drop(columns=["time_key"])
        file_path = os.path.join(queue_dir, f"{time_key}.csv")
        clean_group.to_csv(file_path, index=False)

        count += 1
        if count % 1000 == 0 or count == total_groups:
            print(f"Génération : {count}/{total_groups} fichiers écrits...")

    print("\n--- CRÉATION TERMINÉE AVEC SUCCÈS ---")
    print(f"Dossier {queue_dir} contient maintenant les fichiers de simulation.")


if __name__ == "__main__":
    main()
