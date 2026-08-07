import pandas as pd
from feast import FeatureStore


def run_test():
    # Initialisation du Feature Store en pointant vers le répertoire courant contenant feature_store.yaml
    store = FeatureStore(repo_path=".")

    # Définition d'un exemple d'entités avec leurs caractéristiques requises en entrée
    entity_rows = [
        {
            "cc_num": "4229733778084049",  # Card number present in database
            "amt": 85.50,
            "category": "gas_transport",
            "trans_date_trans_time": "2020-07-22 14:05:00",
            "lat": 45.764043,
            "long": 4.835659,
            "merch_lat": 45.768000,
            "merch_long": 4.840000,
        },
        {
            "cc_num": "9999999999999999",  # Non-existent card number to verify default behavior
            "amt": 50.00,
            "category": "shopping_net",
            "trans_date_trans_time": "2020-07-22 14:05:00",
            "lat": 45.764043,
            "long": 4.835659,
            "merch_lat": 45.768000,
            "merch_long": 4.840000,
        },
    ]

    # Liste des caractéristiques à récupérer
    features_to_fetch = [
        "card_user_features:gender",
        "card_user_features:dob",
        "card_user_features:city_pop",
        "get_derived_features:distance_achat",
        "get_derived_features:age",
        "get_derived_features:hour_sin",
        "get_derived_features:hour_cos",
    ]

    print("Récupération des features en ligne...")
    try:
        response = store.get_online_features(
            features=features_to_fetch, entity_rows=entity_rows
        ).to_dict()

        print("\n[SUCCÈS] Connexion et récupération réussies !")
        df = pd.DataFrame(response)
        print("\nDonnées récupérées :")
        print(df.to_string(index=False))
    except Exception as e:
        print("\n[ERREUR] Impossible de récupérer les features :")
        print(e)


if __name__ == "__main__":
    run_test()
