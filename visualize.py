from src.data_loader import load_data
from src.preprocessing import preprocess
from src.visualization import visualize_before_after


# Colonnes d'origine du CSV


# 1. Charger les données
train_df, _ = load_data()
features = ['Survived','Pclass', 'Sex', 
            'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
            'FamilySize', 'IsAlone', 'AgeGroup', 'FareGroup']
# 2. Garder une copie brute
train_raw = train_df.copy()


# 3. Appliquer le prétraitement
train_cleaned = preprocess(train_df)
train_cleaned = train_cleaned[features]
# Afficher les premières lignes du DataFrame nettoyé
print("\n=== Aperçu des données nettoyées ===")
print(train_cleaned.head())

# 4. Visualisation
visualize_before_after(train_raw, train_cleaned)

