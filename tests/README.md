# 🧪 Stratégie de Test et CI/CD

Ce dossier contient les tests automatisés du projet de détection de fraude. Ces tests sont exécutés automatiquement à chaque étape du pipeline de **CI/CD (Continuous Integration)** pour valider la qualité du code et le bon fonctionnement de l'application avant son déploiement.

---

## 📋 Types de Tests Implémentés

### 1. Tests Unitaires (`pytest`)
Ils vérifient le bon comportement des fonctions isolées (sans base de données ni réseau) :
* **Feature Engineering** :
  * Validation des encodages trigonométriques cycliques (Sinus/Cosinus pour les heures, jours et mois).
  * Validation des calculs géographiques (distance de Haversine entre l'acheteur et le marchand).
  * Validation des calculs d'âge à partir de la date de naissance (`dob`).
* **Validation de Schémas (Pydantic)** :
  * Validation de la structure des données d'entrée (`TransactionInput`).
  * Vérification des contraintes de validation (ex: montant strictement supérieur à 0, format pays).

### 2. Tests d'Intégration
Ils vérifient la communication entre les différents modules de l'application :
* **API FastAPI (`TestClient`)** :
  * Test de la route `/predict` : envoi d'une transaction et vérification de la réponse JSON + code HTTP 200.
  * Test de la route `/reload-model` : vérification de la capacité de l'API à recharger le modèle à la demande.
* **Logs & Ingestion** :
  * Vérification de l'insertion asynchrone des transactions et des scores de fraude (simulée via Mock).
* **Détection de Drift (Evidently AI)** :
  * Exécution de l'analyse de dérive sur des données factices et validation de la structure du rapport JSON produit.

---

## 🛠️ Comment exécuter les tests localement

### Prérequis
Assurez-vous d'avoir installé les dépendances de test :
```bash
pip install -r requirements.txt
```

### Lancer la suite de tests complète
Exécutez la commande suivante à la racine du projet :
```bash
pytest tests/
```

### Lancer l'analyse statique et de style (Linter)
```bash
# Vérification du code avec Ruff
ruff check src/ tests/

# Formatage automatique
ruff format src/ tests/
```
