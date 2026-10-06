# src/utils/api_reloader.py
"""
Gestionnaire de rechargement à chaud (Hot-Reload) de l'API de serving d'inférence.
Permet d'appliquer le modèle Champion nouvellement promu sans interruption de service.
"""

from __future__ import annotations

import os

import requests


def reload_serving_api(custom_url: str | None = None, timeout: int = 5) -> bool:
    """
    Envoie une requête POST pour forcer l'API FastAPI à recharger le modèle @champion depuis MLflow.
    """
    candidate_urls: list[str] = []
    if custom_url:
        candidate_urls.append(custom_url)
    if os.getenv("FASTAPI_RELOAD_URL"):
        candidate_urls.append(os.getenv("FASTAPI_RELOAD_URL"))

    default_urls = [
        "http://ray-head:8000/reload-model",
        "http://localhost:8000/reload-model",
        "http://127.0.0.1:8000/reload-model",
        "http://ray-head:8001/reload-model",
        "http://localhost:8001/reload-model",
    ]
    for u in default_urls:
        if u not in candidate_urls:
            candidate_urls.append(u)

    print("\n🔄 Déclenchement du rechargement à chaud de l'API d'inférence...")
    for url in candidate_urls:
        try:
            resp = requests.post(url, timeout=timeout)
            if resp.status_code == 200:
                print(f"✅ Rechargement à chaud réussi sur {url} : {resp.json()}")
                return True
            else:
                print(f"⚠️ Réponse HTTP {resp.status_code} sur {url} : {resp.text}")
        except requests.RequestException:
            pass

    print(
        "ℹ️ L'API d'inférence n'a pas répondu ou n'est pas active sur les ports configurés."
    )
    return False


# Alias pour rétrocompatibilité et flexibilité
trigger_api_reload = reload_serving_api
