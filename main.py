import os
import sklearn.model_selection
from module3 import main as augment_data
from module1 import extract_features, apply_pca, plot_pca, train_and_evaluate_models, visualize_results
from module2 import main as train_best_model

# Définition des chemins
BASE_DIR = "dataset"
DATASET_DIR = "data_augmented"
FEATURES_FILE = "model_out/features.pkl"
RESULTS_DIR1 = "model_out"
RESULTS_DIR2 = "train_out"

def main():
    """Orchestre les trois étapes : augmentation, extraction/comparaison, optimisation."""
    print("Étape 1 : Augmentation des données")
    augment_data()

    print("\nÉtape 2 : Extraction des caractéristiques et comparaison initiale")
    features, labels = extract_features(DATASET_DIR, FEATURES_FILE)
    X_pca, n_components = apply_pca(features)
    plot_pca(X_pca, labels)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_pca, labels, test_size=0.3, random_state=42)
    results, accuracies = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    visualize_results(results)

    print("\nÉtape 3 : Optimisation et entraînement du meilleur modèle")
    train_best_model(X_train, X_test, y_train, y_test, accuracies)

    print("\nPipeline terminé avec succès.")

if __name__ == "__main__":
    main()