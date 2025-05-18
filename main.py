import pandas as pd
from src.data_loader import load_data
from src.preprocessing import preprocess
from src.model import  predict_model
from src.model import train_model_with_tuning
from imblearn.over_sampling import SMOTE

# 1. Charger les données
train_df, test_df = load_data()

# 2. Prétraitement
train_df = preprocess(train_df, is_test=False)
test_df = preprocess(test_df, is_test=True)

# 3. Sélection des colonnes
features = [
    'Pclass', 'Sex', 'RareTitle', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare',
    'FamilySize', 'IsAlone', 'AgeGroup', 'FareGroup',
    'CabinKnown', 'FarePerPerson', 'Mother', 'Child'
]

X = train_df[features]
y = train_df['Survived']
X_test = test_df[features]


# Appliquer SMOTE ici
#sm = SMOTE(random_state=42 , sampling_strategy ='auto', k_neighbors=5)
#X_res, y_res = sm.fit_resample(X, y)
#print("=== avant SMOTE ===")
#print("Taille de X :", X.shape)
#print("Taille de y :", y.shape)
#print("Distribution de y :")
#print(y.value_counts(normalize=True))

#print("=== Après SMOTE ===")
#print("Taille de X_res :", X_res.shape)
#print("Taille de y_res :", y_res.shape) 
#print("Distribution de y_res :")
#print(y_res.value_counts(normalize=True))

# Puis entraîner le modèle sur X_res, y_res 
#model = train_model_with_tuning(X_res, y_res)

# 5. Entraînement
model = train_model_with_tuning(X, y)

# 6. Prédiction
predictions = predict_model(model, X_test)

# 7. Export pour soumission Kaggle
output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predictions})
if len(output) != len(test_df):
    raise ValueError("Le nombre de lignes dans le DataFrame de sortie ne correspond pas à celui du DataFrame de test.")
output.to_csv('submission.csv', index=False)
print("✅ Fichier submission.csv créé avec succès.")
