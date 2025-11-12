### README.md complet

```markdown
# 🥬 Analyseur de Têtê : Classification IA des Légumes

[![Streamlit](https://img.shields.io/badge/Streamlit-FF6B35?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6B35?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)

## Description
L'Analyseur de Têtê est une application IA qui classe les images de Têtê (un légume spécifique) en "Bonne qualité" ou "Mauvaise qualité", et détecte si ce n'est pas un Têtê. Basé sur MobileNetV2 pour l'extraction de caractéristiques, PCA pour la réduction, et XGBoost optimisé pour la classification. Interface web simple avec Streamlit pour uploader une photo ou utiliser la webcam.

**Objectif** : Aider les utilisateurs à évaluer rapidement la qualité des Têtês au marché ou en cuisine.

**Démos** : [Lien Streamlit déployé]([https://tonapp.streamlit.app](https://vegetablesimagesclassification-xdvkukq6hw3lsifp94pjt6.streamlit.app/).

## Installation
1. **Clone le repo** :
   ```
   git clone https://github.com/nuvnce/Vegetables_images_classification.git
   cd Vegetables_images_classification
   ```

2. **Environnement virtuel (recommandé)** :
   ```
   python -m venv env
   source env/bin/activate  # Linux/Mac
   # ou env\Scripts\activate  # Windows
   ```

3. **Dépendances** :
   ```
   pip install -r requirements.txt
   ```

4. **Données** :
   - Téléchargez les images originales depuis [ce lien Drive](https://drive.google.com/drive/folders/1GWlKu86ZhXhCpssvtdfePka39P0y3xL9?usp=sharing).
   - Extrayez dans `dataset/` (structure : `bon/` et `mauvais/`).

5. **Génération des modèles** :
   ```
   python main.py
   ```
   - Ça crée `data_augmented/` (augmentées), `model_out/` (extraction/PCA), et `train_out/` (modèle optimisé).

## Utilisation
1. **Pipeline complet** (une fois) :
   ```
   python main.py
   ```
   - Augmente les données, extrait les features, compare et optimise les modèles.

2. **Interface** :
   ```
   streamlit run app.py
   ```
   - Ouvrez `http://localhost:8501`.
   - Chargez une image ou prenez une photo.
   - Cliquez "Prédire" pour le verdict (ex. "Ce Têtê est de bonne qualité !").

**Exemples de sortie** :
- Bonne qualité : 92 % confiance.
- Mauvaise qualité : 87 % confiance.
- Non-Têtê : Avertissement si < 70 % confiance.

## Structure du projet
```
classification/
├── dataset/          # Images originales (télécharger depuis Drive)
│   ├── bon/
│   └── mauvais/
├── data_augmented/           # Images augmentées (générées par main.py)
├── model_out/        # Features, PCA, modèles initiaux
├── train_out/        # Modèle optimisé final
├── app.py            # Interface Streamlit
├── main.py           # Orchestration pipeline
├── module1.py  # Extraction & comparaison
├── module2.py  # Optimisation
├── module3.py  # Augmentation
├── requirements.txt  # Dépendances
└── README.md         # Ce fichier
```

## Résultats
- **Précision** : 90 % sur jeu de test.
- **AUC** : 0.90 (courbe ROC).
- **Graphiques** : Voir `model_out/` et `train_out/` (PCA, matrices de confusion, ROC).

Consultez le [rapport complet](https://drive.google.com/file/d/1ByggdRM5Eflu7oTJeRxax4x6sk-YwEUx/view?usp=sharing) pour détails techniques et évaluation.

## Dépendances
Voir `requirements.txt` :
- TensorFlow, NumPy, Matplotlib, Scikit-learn, XGBoost, Streamlit, etc.

## Licence
MIT License – Utilisez librement, citez-moi si possible !

## Contributeurs
- [Daniel ESSONANI] – Développement principal.

## Support
Issues sur GitHub ou contacte-moi. Amuse-toi bien avec tes Têtês ! 🥬
```
