import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tqdm import tqdm  # Pour la barre de progression

# Définition des chemins
FEATURES_FILE = "model_out/features.pkl"
RESULTS_DIR = "train_out"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_features(features_file):
    """Charge les caractéristiques et labels depuis un fichier."""
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Le fichier {features_file} n'existe pas.")
    with open(features_file, 'rb') as f:
        data = pickle.load(f)
    return data['features'], data['labels']


def apply_pca(X, variance_threshold=0.95):
    """Réduit la dimensionnalité avec un seuil de variance plus strict."""
    pca = PCA()
    X_pca = pca.fit_transform(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_optimal = np.argmax(cumulative_variance >= variance_threshold) + 1
    print(f"Nombre optimal de composantes PCA: {n_components_optimal}")
    return PCA(n_components=n_components_optimal).fit_transform(X), n_components_optimal


def optimize_model(model_name, X_train, y_train):
    """Optimise le modèle sélectionné avec une recherche aléatoire d'hyperparamètres."""
    if model_name == "SVM":
        model = SVC(probability=True, random_state=42)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'kernel': ['linear', 'rbf', 'poly'],
            'class_weight': ['balanced', None]
        }
    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', None]
        }
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.7, 0.9, 1.0]
        }
    else:
        raise ValueError(f"Modèle {model_name} non supporté.")

    # Recherche aléatoire avec feedback
    grid_search = RandomizedSearchCV(
        model, param_grid, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    print(f"Optimisation de {model_name} en cours...")
    grid_search.fit(X_train, y_train)
    print(f"Meilleurs paramètres pour {model_name}: {grid_search.best_params_}")
    return grid_search.best_estimator_


def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Évalue le modèle optimisé et compare avec une version par défaut."""
    if model_name == "SVM":
        default_model = SVC(probability=True, random_state=42)
    elif model_name == "Random Forest":
        default_model = RandomForestClassifier(random_state=42)
    elif model_name == "XGBoost":
        default_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    with tqdm(total=1, desc=f"Entraînement {model_name} par défaut") as pbar:
        default_model.fit(X_train, y_train)
        pbar.update(1)
    default_pred = default_model.predict(X_test)
    print(f"\n{model_name} par défaut - Rapport de classification:\n", classification_report(y_test, default_pred))

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = auc(fpr, tpr)

    print(f"\n{model_name} optimisé - Rapport de classification:\n", classification_report(y_test, y_pred))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Matrice de confusion - {model_name} optimisé")
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies valeurs")
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"))
    plt.close()

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f"roc_curve_{model_name.lower().replace(' ', '_')}.png"))
    plt.close()

    return report, cm, auc_score


def main(X_train, X_test, y_train, y_test, accuracies):
    """Optimise et entraîne le meilleur modèle basé sur les accuracies du module 1."""
    best_model_name = max(accuracies, key=accuracies.get)
    print(f"\nMeilleur modèle sélectionné: {best_model_name} avec accuracy {accuracies[best_model_name]:.4f}")

    best_model = optimize_model(best_model_name, X_train, y_train)
    results = evaluate_model(best_model, best_model_name, X_train, X_test, y_train, y_test)

    model_path = os.path.join(RESULTS_DIR, f"{best_model_name.lower().replace(' ', '_')}_best_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Modèle {best_model_name} optimisé sauvegardé dans {model_path}")
