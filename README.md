# Titanic - Machine Learning from Disaster

Ce projet Kaggle prédit la survie des passagers du Titanic à partir de données tabulaires.

## Structure du projet

- `src/`
  - `data_loader.py` : Chargement des données
  - `preprocessing.py` : Prétraitement et feature engineering
  - `model.py` : Entraînement et sélection de modèles (RandomForest, XGBoost, VotingClassifier, etc.)
  - `utils.py` : Utilitaires (création d'environnement, installation)
  - `visualization.py` : Fonctions de visualisation
- `main.py` : Pipeline en script (prétraitement, entraînement, prédiction, export Kaggle) 
- `notebook.py` : Pipeline notebook avec visu (prétraitement, entraînement, prédiction, export Kaggle)
- `requirements.txt` : Dépendances Python
- `README.md` : Ce fichier

## Modèles disponibles
`main.py`
- RandomForest (avec grid search)
- XGBoost (avec grid search)
- VotingClassifier (ensemble)
- Sélection de variables (Recursive Feature Elimination)
- Visualisation des données



`notebook.py`
- Travail sur Features plus précis
- RandomForest (avec grid search)
- GradientBoosting (avec grid search)
- LogisticRegression (avec grid search)
- XGBoost (avec grid search)
- Test avec cross_validation
- StackingClassifier avec plusieurs combinaisons
-  Visualisation des données

## Score

- `Main.py` Best Score: 0.77751
- `notebook.py`  Best Score: 0.78229

## Utilisation

Lancez le pipeline complet :
```sh
python main.py
```
Le fichier `submission.csv` sera généré pour soumission Kaggle.
