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
        st.error(f"Aucun modèle optimisé trouvé dans {results_dir}. Veuillez exécuter le pipeline d'entraînement d'abord.")
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

# CSS personnalisé amélioré
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f0f2f6 0%, #e0e7ff 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: bold;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #45a049, #4CAF50);
        transform: translateY(-1px);
    }
    .title {
        color: #2E7D32;
        font-size: 40px;
        text-align: center;
        font-family: 'Arial Black', sans-serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        color: #388E3C;
        font-size: 22px;
        text-align: center;
        font-style: italic;
    }
    .result-good {
        color: #4CAF50;
        font-size: 28px;
        text-align: center;
        font-weight: bold;
    }
    .result-bad {
        color: #F44336;
        font-size: 28px;
        text-align: center;
        font-weight: bold;
    }
    .warning-text {
        font-size: 24px;
        text-align: center;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Fonction principale
def main():
    # Sidebar pour infos
    with st.sidebar:
        st.title("📊 Infos Modèle")
        st.write("**Modèle :** XGBoost optimisé")
        st.write("**Précision :** 90%")
        st.write("**AUC :** 0.90")
        st.image("predicts_img/pca_plot.png", caption="Séparation PCA", use_container_width=True)

    # Initialisation de l'état
    if 'intro_accepted' not in st.session_state:
        st.session_state.intro_accepted = False

    # Page d'introduction
    if not st.session_state.intro_accepted:
        st.markdown("<h1 class='title'>Bienvenue à l'Analyseur de Têtê 🌿</h1>", unsafe_allow_html=True)
        st.markdown("""
            <p style='text-align: center; font-size: 18px; margin: 20px 0;'>
            Cette application utilise l'intelligence artificielle pour déterminer si un Têtê est de bonne ou mauvaise qualité. 
            Téléchargez une image ou prenez une photo, et notre modèle vous dira tout sur votre Têtê ! 
            Prêt à explorer ? Cochez la case ci-dessous pour commencer.
            </p>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🎯 Commencer l'aventure", key="start_btn"):
                st.session_state.intro_accepted = True
                st.rerun()
        with col2:
            st.image("data_augmented/augmentation_sample.png", caption="Exemple d'augmentation", use_container_width=True)

        return

    # Interface principale
    st.markdown("<h1 class='title'>Analyseur de Têtê 🌱</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Vérifiez la qualité de vos Têtês en un clin d'œil !</p>", unsafe_allow_html=True)

    # Charger les modèles
    pca_model = load_pca_model(PCA_MODEL_PATH)
    best_model, model_name = load_best_model(RESULTS_DIR)

    if pca_model is None or best_model is None:
        st.stop()

    # Sélection de l'option
    option = st.radio("🔍 Choisissez une méthode :", ("Charger une image", "Prendre une photo"))

    image = None
    if option == "Charger une image":
        uploaded_file = st.file_uploader("📁 Choisissez une image de Têtê...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
    else:
        camera_file = st.camera_input("📸 Prenez une photo de votre Têtê")
        if camera_file is not None:
            image = Image.open(camera_file).convert('RGB')

    if image is not None:
        container = st.container()
        with container:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(image, caption="Votre Têtê", use_container_width=True)
            with col2:
                if st.button("🔮 Prédire"):
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
                            st.markdown("<p class='warning-text'>⚠️ Il ne s’agit pas de Têtê ! La prédiction est trop incertaine.</p>", unsafe_allow_html=True)
                            st.balloons()  # Animation fun
                        else:
                            if prediction == 0:
                                st.markdown("<p class='result-good'>Ce Têtê est de bonne qualité ! 🌟</p>", unsafe_allow_html=True)
                                st.balloons(color="green")
                            else:
                                st.markdown("<p class='result-bad'>Ce Têtê est de mauvaise qualité... 😞</p>", unsafe_allow_html=True)
                                st.balloons(color="red")
                            st.metric("Confiance", f"{max_prob:.2f}%", delta=None)

                            # Probabilités détaillées
                            col_prob1, col_prob2 = st.columns(2)
                            with col_prob1:
                                st.metric("Bonne qualité", f"{probability[0] * 100:.2f}%")
                            with col_prob2:
                                st.metric("Mauvaise qualité", f"{probability[1] * 100:.2f}%")

    # Pied de page
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666; font-size: 14px;'>Créé avec ❤️ pour les amoureux de Têtê</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()