# Titanic - Machine Learning from Disaster

Ce projet Kaggle prédit la survie des passagers du Titanic à partir de données tabulaires.

## Structure du projet

- `src/`
  - `data_loader.py` : Chargement des données
  - `preprocessing.py` : Prétraitement et feature engineering
  - `model.py` : Entraînement et sélection de modèles (RandomForest, XGBoost, VotingClassifier, etc.)
  - `utils.py` : Utilitaires (création d'environnement, installation)
  - `visualization.py` : Fonctions de visualisation
- `main.py` : Pipeline principal (prétraitement, entraînement, prédiction, export Kaggle)
- `requirements.txt` : Dépendances Python
- `README.md` : Ce fichier

## Modèles disponibles

- RandomForest (avec grid search)
- XGBoost (avec grid search)
- VotingClassifier (ensemble)
- Sélection de variables (RFE)
- Visualisation des données

## Améliorations possibles

- Ajouter des notebooks d’exploration (`.ipynb`)
- Ajouter des tests unitaires (`pytest`)
- Ajouter un script d’évaluation automatique sur validation locale

## Utilisation

Lancez le pipeline complet :
```sh
python main.py
```
Le fichier `submission.csv` sera généré pour soumission Kaggle.
