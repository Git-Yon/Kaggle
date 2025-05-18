from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from tqdm import tqdm
import sys
from xgboost import XGBClassifier

# Progress bar pour GridSearchCV
class TqdmGridSearchCV(GridSearchCV):
    def fit(self, X, y=None, **fit_params):
        total = 1
        for v in self.param_grid.values():
            total *= len(v)
        with tqdm(total=total, desc="GridSearchCV", file=sys.stdout) as pbar:
            self._pbar = pbar
            return super().fit(X, y, **fit_params)
    def _run_search(self, evaluate_candidates):
        def wrapper(candidate_params):
            self._pbar.update(len(candidate_params))
            return evaluate_candidates(candidate_params)
        super()._run_search(wrapper)

# MODELE 1 : RandomForestClassifier avec grid search complet
def train_rf_full_grid(X, y):
    rf = RandomForestClassifier(random_state=42, bootstrap=True, class_weight='balanced_subsample')
    param_grid = {
        'n_estimators': [1000, 2000],
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],
        'min_samples_leaf': [1, 2, 4],
        'max_samples': [0.5, 0.75, 1.0],
        'max_depth': [None, 10, 50],
        'min_samples_split': [2, 5, 10, 20]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    print("Meilleurs paramètres :", grid_search.best_params_)
    print("Meilleur score CV :", grid_search.best_score_)
    return grid_search.best_estimator_

# MODELE 2 : RandomForestClassifier avec SMOTE et grid search sur n_estimators
def train_rf_smote(X, y):
    rf = RandomForestClassifier(
        random_state=42,
        bootstrap=True,
        class_weight='balanced_subsample',
        max_depth=None,
        max_features='sqrt',
        max_samples=1.0,
        min_samples_leaf=4,
        min_samples_split=2,
        criterion='gini'
    )
    param_grid = {
        'n_estimators': [100, 500, 1000, 2000]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    print("Meilleurs paramètres :", grid_search.best_params_)
    print("Meilleur score CV :", grid_search.best_score_)
    return grid_search.best_estimator_

# MODELE 3 : RandomForestClassifier + RFE (feature selection)
def train_rf_rfe(X, y, n_features_to_select=8):
    base_rf = RandomForestClassifier(
        random_state=42,
        bootstrap=True,
        class_weight='balanced_subsample',
        max_depth=None,
        max_features='sqrt',
        max_samples=1.0,
        min_samples_leaf=4,
        min_samples_split=2,
        criterion='gini',
        n_estimators=200
    )
    selector = RFE(base_rf, n_features_to_select=n_features_to_select)
    selector.fit(X, y)
    X_selected = selector.transform(X)
    print("Features sélectionnées :", list(X.columns[selector.support_]))

    rf = RandomForestClassifier(
        random_state=42,
        bootstrap=True,
        class_weight='balanced_subsample',
        max_depth=None,
        max_features='sqrt',
        max_samples=1.0,
        min_samples_leaf=4,
        min_samples_split=2,
        criterion='gini'
    )
    param_grid = {
        'n_estimators': [100, 500, 1000, 2000]
    }
    grid_search = TqdmGridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_selected, y)
    print("Meilleurs paramètres :", grid_search.best_params_)
    print("Meilleur score CV :", grid_search.best_score_)
    return grid_search.best_estimator_, selector

def predict_model_rfe(model_selector_tuple, X_test):
    model, selector = model_selector_tuple
    X_test_selected = selector.transform(X_test)
    return model.predict(X_test_selected)

# MODELE 4 : VotingClassifier (ensemble de plusieurs modèles)
def train_voting_classifier(X, y):
    clf1 = LogisticRegression(max_iter=1000, random_state=42)
    clf2 = RandomForestClassifier(
        random_state=42,
        bootstrap=True,
        class_weight='balanced_subsample',
        max_depth=None,
        max_features='sqrt',
        max_samples=1.0,
        min_samples_leaf=4,
        min_samples_split=2,
        criterion='gini',
        n_estimators=500
    )
    clf3 = GradientBoostingClassifier(n_estimators=200, random_state=42)
    clf4 = SVC(probability=True, random_state=42)

    voting_clf = VotingClassifier(
        estimators=[
            ('lr', clf1),
            ('rf', clf2),
            ('gb', clf3),
            ('svc', clf4)
        ],
        voting='soft',
        n_jobs=-1
    )
    voting_clf.fit(X, y)
    print("VotingClassifier entraîné avec succès.")
    return voting_clf

# MODELE 5 : XGBoost avec grid search sur n_estimators et learning_rate
def train_xgb_grid(X, y):
    param_grid = {
        'n_estimators': [100, 200, 500, 1000],
        'learning_rate': [0.05, 0.1, 0.5, 1.0],
    }
    model = XGBClassifier(
        max_depth=4,
        subsample=0.5,
        colsample_bytree=0.5,
        random_state=42,
        eval_metric='logloss'
    )
    grid_search = TqdmGridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X, y)
    print("Meilleurs paramètres :", grid_search.best_params_)
    print("Meilleur score CV :", grid_search.best_score_)
    return grid_search.best_estimator_











































# Fonction générique de prédiction
def predict_model(model, X_test):
    return model.predict(X_test)