# 🧪 Stratégie de Test et CI/CD

Ce dossier contient les tests automatisés du projet de détection de fraude. Ces tests sont exécutés automatiquement à chaque étape du pipeline de **CI/CD (Continuous Integration)** pour valider la qualité du code et le bon fonctionnement de l'application avant son déploiement.

---

## 📋 Tests Implémentés

La suite de tests contient **4 tests clés** validant le pipeline de bout en bout :

### 1. Tests Unitaires (`pytest`)
Ils vérifient le bon comportement des fonctions mathématiques et utilitaires isolées :
* **`test_haversine_distance`** : Validation du calcul géographique de la distance de Haversine (ex: calcul de la distance orthodromique réelle Lyon-Paris, attendue autour de ~391 km).
* **`test_cyclical_inverse_retransformation`** : Vérification des transformations trigonométriques inverses (sinus/cosinus) avec l'évaluation de l'heure en sortie de l'ingénierie des caractéristiques.

### 2. Tests d'Intégration API (`fastapi.testclient`)
Ils vérifient le comportement des routes de l'API et la logique de suspicion :
* **`test_api_health`** : Vérification de la route de diagnostic `/health` (renvoie `200 OK` et `{"status": "healthy"}`).
* **`test_api_predict_batch_schema`** : Envoi d'un batch de transactions suspects à la route `/predict_batch` et validation :
  * Du bon typage et de la structure de la réponse JSON.
  * Du déclenchement correct des règles du **Fast Pass Redis** (suspicion = 1).

---

## 🔒 Architecture des Mocks en Intégration Continue (CI/CD)

Pour garantir que les tests s'exécutent en moins de **3 secondes** sur les serveurs de CI/CD (GitHub Actions) sans nécessiter une infrastructure lourde (base de données, serveur MLflow, cache Redis, ou compilations C++ de SHAP), les composants externes suivants sont intégralement **mockés** au démarrage des tests :

* **MLflow / MLflow.sklearn** : Court-circuités pour simuler le chargement d'un modèle champion fictif sans connexion réseau.
* **Redis** : Simulé via un client Mock renvoyant des règles de suspicion configurées en mémoire.
* **SHAP** : Le package de calcul d'explicabilité locale est mocké pour éviter d'installer des paquets binaires lourds dans l'environnement temporaire de CI.
* **SQLAlchemy** : Mocké pour bloquer les tentatives de chargement du pilote dialecte PostgreSQL (`psycopg2`) et accélérer le chargement de l'API.

---

## 🛠️ Comment exécuter les tests localement

### Prérequis
Vous devez disposer de l'environnement Python du projet ou utiliser l'outil `uv` (recommandé) :
```bash
pip install -r requirements.txt
```

### Lancer la suite de tests complète
Exécutez la commande suivante à la racine du projet :
```bash
# Avec uv (recommandé)
uv run pytest tests/

# Ou avec pytest classique
pytest tests/
```

### Lancer l'analyse statique et de style (Linter)
```bash
# Vérification du code avec Ruff
uv run ruff check .

# Formatage automatique
uv run ruff format .
```
