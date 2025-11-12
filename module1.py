import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tqdm import tqdm  # Pour la barre de progression

# Définition des chemins
DATASET_DIR = "data_augmented"
FEATURES_FILE = "model_out/features.pkl"
RESULTS_DIR = "model_out"
os.makedirs(RESULTS_DIR, exist_ok=True)


def extract_features(dataset_dir, features_file):
    """Extrait les caractéristiques des images dans les sous-dossiers de dataug/ avec progression."""
    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
        raise ValueError(f"Le dossier {dataset_dir} est vide ou n'existe pas.")

    # Collecte des images
    image_paths = []
    labels = []
    for class_name in ['bon', 'mauvais']:
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        for sub_dir in os.listdir(class_dir):
            sub_dir_path = os.path.join(class_dir, sub_dir)
            if os.path.isdir(sub_dir_path):
                for img_file in os.listdir(sub_dir_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(sub_dir_path, img_file))
                        labels.append(0 if class_name == 'bon' else 1)

    if not image_paths:
        raise ValueError("Aucune image trouvée dans les sous-dossiers de dataug/.")

    # Charge MobileNetV2
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

    # Extraction des caractéristiques avec barre de progression
    features = []
    for img_path in tqdm(image_paths, desc="Extraction des caractéristiques"):
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        feature = model.predict(img_array, verbose=0)
        features.append(feature.flatten())

    features = np.array(features)
    labels = np.array(labels)

    # Sauvegarde
    os.makedirs(os.path.dirname(features_file), exist_ok=True)
    with open(features_file, 'wb') as f:
        pickle.dump({'features': features, 'labels': labels}, f)
    print(f"Extraction terminée. Caractéristiques sauvegardées dans {features_file}")
    return features, labels


def load_features(features_file):
    """Charge les caractéristiques et labels depuis un fichier."""
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Le fichier {features_file} n'existe pas.")
    with open(features_file, 'rb') as f:
        data = pickle.load(f)
    return data['features'], data['labels']


def apply_pca(X, variance_threshold=0.90):
    """Réduit la dimensionnalité des caractéristiques avec PCA."""
    pca = PCA()
    X_pca = pca.fit_transform(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_optimal = np.argmax(cumulative_variance >= variance_threshold) + 1
    print(f"Nombre optimal de composantes PCA: {n_components_optimal}")

    pca_optimal = PCA(n_components=n_components_optimal)
    X_pca = pca_optimal.fit_transform(X)
    with open(os.path.join(RESULTS_DIR, "pca_model.pkl"), 'wb') as f:
        pickle.dump(pca_optimal, f)
    return X_pca, n_components_optimal


def plot_pca(X_pca, y):
    """Visualise les deux premières composantes PCA."""
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
    plt.title("Projection PCA (2 composantes)")
    plt.xlabel("Composante 1")
    plt.ylabel("Composante 2")
    plt.savefig(os.path.join(RESULTS_DIR, "pca_plot.png"))
    plt.close()


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Entraîne et évalue plusieurs modèles avec validation croisée et progression."""
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "SVM": SVC(probability=True, kernel='rbf', random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=6, random_state=42)
    }
    results = {}
    accuracies = {}

    for name, model in models.items():
        print(f"\nEntraînement de {name}...")
        scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"{name} - Cross-validation accuracy: {scores.mean():.4f} (±{scores.std():.4f})")

        with tqdm(total=1, desc=f"Entraînement {name}") as pbar:
            model.fit(X_train, y_train)
            pbar.update(1)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model": model,
            "report": report,
            "confusion_matrix": cm,
            "roc_curve": (roc_curve(y_test, y_proba), auc(roc_curve(y_test, y_proba)[0], roc_curve(y_test, y_proba)[1]))
        }
        accuracies[name] = report['accuracy']

        print(f"\n{name} - Accuracy: {report['accuracy']:.4f}")
        print(classification_report(y_test, y_pred))

        model_path = os.path.join(RESULTS_DIR, f"{name.lower().replace(' ', '_')}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Modèle {name} sauvegardé dans {model_path}")
    return results, accuracies


def visualize_results(results):
    """Génère des visualisations pour chaque modèle."""
    for name, res in results.items():
        plt.figure(figsize=(6, 5))
        sns.heatmap(res["confusion_matrix"], annot=True, fmt='d', cmap='Blues')
        plt.title(f"Matrice de confusion - {name}")
        plt.xlabel("Prédictions")
        plt.ylabel("Vraies valeurs")
        plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{name.lower().replace(' ', '_')}.png"))
        plt.close()

        fpr, tpr, _ = res["roc_curve"][0]
        auc_score = res["roc_curve"][1]
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve - {name} (AUC = {auc_score:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, f"roc_curve_{name.lower().replace(' ', '_')}.png"))
        plt.close()


def main():
    """Exécute l'extraction, la comparaison des modèles et retourne les données."""
    features, labels = extract_features(DATASET_DIR, FEATURES_FILE)
    X_pca, n_components = apply_pca(features)
    plot_pca(X_pca, labels)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, labels, test_size=0.3, random_state=42)
    results, accuracies = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    visualize_results(results)
    print("Comparaison des modèles terminée.")
    return X_train, X_test, y_train, y_test, accuracies


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, accuracies = main()