import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_before_after(before_df, after_df):
    # === Infos générales ===
    print("=== AVANT PRÉTRAITEMENT ===")
    print(before_df.info())
    print(before_df.isnull().sum())

    print("\n=== APRÈS PRÉTRAITEMENT ===")
    print(after_df.info())
    print(after_df.isnull().sum())

    # === Valeurs manquantes ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(before_df.isnull(), cbar=False, yticklabels=False, cmap="viridis", ax=axes[0])
    axes[0].set_title("Valeurs manquantes - Avant")

    sns.heatmap(after_df.isnull(), cbar=False, yticklabels=False, cmap="viridis", ax=axes[1])
    axes[1].set_title("Valeurs manquantes - Après")
    plt.tight_layout()
    plt.show()

    # === Distribution de l’âge ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(before_df['Age'], bins=20, kde=True, ax=axes[0])
    axes[0].set_title('Age - Avant')

    sns.histplot(after_df['Age'], bins=20, kde=True, ax=axes[1])
    axes[1].set_title('Age - Après')
    plt.tight_layout()
    plt.show()

    # === Sexe ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(x='Sex', data=before_df, ax=axes[0])
    axes[0].set_title('Sexe - Avant')

    sns.countplot(x='Sex', data=after_df, ax=axes[1])
    axes[1].set_title('Sexe - Après')
    plt.tight_layout()
    plt.show()

    # === Heatmap de corrélations ===
    corr = after_df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Corrélation des variables après prétraitement")
    plt.show()

    # === Distribution de la classe ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)) 
    sns.countplot(x='Pclass', data=before_df, ax=axes[0])
    axes[0].set_title('Classe - Avant')
    sns.countplot(x='Pclass', data=after_df, ax=axes[1])
    axes[1].set_title('Classe - Après')
    plt.tight_layout()
    plt.show()


    