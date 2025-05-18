import pandas as pd
import re

def preprocess(df, is_test=False):
    df = df.copy()
    

    # Si tu as déjà extrait Title, sinon adapte avec Sex/Pclass
    # Extraire le titre
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]*)\.', expand=False)
    # Regrouper les titres rares
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    df['RareTitle'] = (df['Title'] == 'Rare').astype(int)
    # Imputer l'âge par la médiane du titre
    df['Age'] = df.groupby(['Sex', 'Pclass', 'Title'])['Age'].transform(lambda x: x.fillna(x.median()))

    # CabinKnown : 1 si la cabine est connue, 0 sinon
    df['CabinKnown'] = (~df['Cabin'].isnull()).astype(int) if 'Cabin' in df.columns else 0

    

    # Mother : femme adulte avec enfants
    df['Mother'] = ((df['Sex'] == 1) & (df['Parch'] > 0) & (df['Age'] > 18) & (df['Title'] == 'Mrs')).astype(int)

    # Child : Age < 12 ou Title == Master
    df['Child'] = ((df['Age'] < 12) | (df['Title'] == 'Master')).astype(int)

    df['Fare'] = df.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    # 3. Encodage du sexe
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # 4. Taille de la famille
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # 5. Est-ce que la personne est seule ?
    df['IsAlone'] = 1
    df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0

    # 6. AgeGroup : 0=Enfant, 1=Ado, 2=Adulte, 3=Senior
    def age_group(age):
        if age < 12:
            return 0  # Enfant
        elif age < 18:
            return 1  # Ado
        elif age < 60:
            return 2  # Adulte
        else:
            return 3  # Senior
    df['AgeGroup'] = df['Age'].apply(age_group)

    # 7. FareGroup : 0=Bas, 1=Moyen, 2=Élevé (par tertiles)
    df['FareGroup'] = pd.qcut(df['Fare'], 3, labels=[0, 1, 2]).astype(int)

    # 8. (Optionnel) Features avancées à décommenter si besoin
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # FarePerPerson
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    # # Extraire le titre depuis le nom
    # df['Title'] = df['Name'].apply(lambda x: re.search(r',\s*([^\.]*)\.', x).group(1).strip())
    # # Regrouper les titres rares
    # df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    # df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    # df['Title'] = df['Title'].replace('Mme', 'Mrs')
    # # Encodage numérique du titre
    # title_mapping = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
    # df['Title'] = df['Title'].map(title_mapping).fillna(4).astype(int)

    # # Ticket features
    # df['TicketLength'] = df['Ticket'].apply(lambda x: len(str(x)))
    # df['TicketPrefix'] = df['Ticket'].apply(lambda x: str(x).replace('.', '').replace('/', '').split()[0] if not str(x).split()[0].isdigit() else 'NONE')
    # df['TicketCount'] = df.groupby('Ticket')['Ticket'].transform('count')
    # # SharedTicket: 1 si plusieurs passagers partagent le même ticket, sinon 0
    # df['SharedTicket'] = (df['TicketCount'] > 1).astype(int)
    # # Encodage du TicketPrefix (Label Encoding)
    # ticket_prefixes = df['TicketPrefix'].unique()
    # prefix_mapping = {prefix: idx for idx, prefix in enumerate(ticket_prefixes)}
    # df['TicketPrefix'] = df['TicketPrefix'].map(prefix_mapping)


    return df