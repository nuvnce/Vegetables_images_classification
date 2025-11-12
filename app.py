import streamlit as st
import os
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# Chemins des fichiers sauvegardés
PCA_MODEL_PATH = "model_out/pca_model.pkl"
RESULTS_DIR = "train_out"


# Fonction pour charger le modèle PCA
def load_pca_model(pca_path):
    if not os.path.exists(pca_path):
        st.error(f"Modèle PCA introuvable à {pca_path}. Veuillez exécuter le pipeline d'entraînement d'abord.")
        return None
    with open(pca_path, 'rb') as f:
        return pickle.load(f)


# Fonction pour charger le modèle optimisé
def load_best_model(results_dir):
    model_files = [f for f in os.listdir(results_dir) if f.endswith('_best_model.pkl')]
    if not model_files:
        st.error(
            f"Aucun modèle optimisé trouvé dans {results_dir}. Veuillez exécuter le pipeline d'entraînement d'abord.")
        return None, None
    model_path = os.path.join(results_dir, model_files[0])
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    model_name = model_files[0].replace('_best_model.pkl', '').replace('_', ' ').title()
    return model, model_name


# Fonction pour extraire les caractéristiques d'une image avec MobileNetV2
def extract_features(image):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    features = model.predict(img_array, verbose=0)
    return features.flatten()


# CSS personnalisé pour améliorer l'apparence
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .title {
        color: #2E7D32;
        font-size: 36px;
        text-align: center;
    }
    .subtitle {
        color: #388E3C;
        font-size: 20px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


# Fonction principale
def main():
    # Initialisation de l'état
    if 'intro_accepted' not in st.session_state:
        st.session_state.intro_accepted = False

    # Page d'introduction
    if not st.session_state.intro_accepted:
        st.markdown("<h1 class='title'>Bienvenue à l'Analyseur de Têtê 🌿</h1>", unsafe_allow_html=True)
        st.markdown("""
            <p style='text-align: center;'>
            Cette application utilise l'intelligence artificielle pour déterminer si un Têtê est de bonne ou mauvaise qualité. 
            Téléchargez une image ou prenez une photo, et notre modèle vous dira tout sur votre Têtê ! 
            Prêt à explorer ? Cochez la case ci-dessous pour commencer.
            </p>
        """, unsafe_allow_html=True)

        if st.checkbox("J'accepte de commencer", key="intro_checkbox"):
            st.session_state.intro_accepted = True
            st.rerun()  # Forcer le rechargement immédiat
        return

    # Interface principale
    st.markdown("<h1 class='title'>Analyseur de Têtê 🌱</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Vérifiez la qualité de vos Têtês en un clin d'œil !</p>", unsafe_allow_html=True)

    # Charger les modèles
    pca_model = load_pca_model(PCA_MODEL_PATH)
    best_model, model_name = load_best_model(RESULTS_DIR)

    if pca_model is None or best_model is None:
        st.stop()

    st.write(f"**Modèle utilisé :** {model_name}")

    # Sélection de l'option
    option = st.radio("Choisissez une méthode :", ("Charger une image", "Prendre une photo"))

    image = None
    if option == "Charger une image":
        uploaded_file = st.file_uploader("Choisissez une image de Têtê...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
    else:
        camera_file = st.camera_input("Prenez une photo de votre Têtê")
        if camera_file is not None:
            image = Image.open(camera_file).convert('RGB')

    if image is not None:
        # Mise en page en colonnes
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, caption="Votre Têtê", use_container_width=True)
        with col2:
            if st.button("Prédire"):
                with st.spinner("Analyse de votre Têtê en cours..."):
                    # Extraire les caractéristiques
                    features = extract_features(image)
                    features_pca = pca_model.transform(features.reshape(1, -1))

                    # Faire la prédiction
                    prediction = best_model.predict(features_pca)[0]
                    probability = best_model.predict_proba(features_pca)[0]
                    max_prob = np.max(probability) * 100
                    diff_prob = abs(probability[0] - probability[1]) * 100

                    # Afficher un message personnalisé
                    if max_prob < 70 or diff_prob < 20:
                        st.warning("⚠️ Il ne s’agit pas de Têtê ! La prédiction est trop incertaine.")
                    else:
                        if prediction == 0:
                            st.markdown("<h3 style='color: #4CAF50;'>Ce Têtê est de bonne qualité ! 🌟</h3>",
                                        unsafe_allow_html=True)
                        else:
                            st.markdown("<h3 style='color: #F44336;'>Ce Têtê est de mauvaise qualité... 😞</h3>",
                                        unsafe_allow_html=True)
                        st.write(f"Confiance : {max_prob:.2f}%")

                        # Probabilités détaillées
                        st.write("**Détails de l’analyse :**")
                        st.write(f"- Bonne qualité : {probability[0] * 100:.2f}%")
                        st.write(f"- Mauvaise qualité : {probability[1] * 100:.2f}%")

    # Pied de page
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Créé avec ❤️ pour les amoureux de Têtê</p>",
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()