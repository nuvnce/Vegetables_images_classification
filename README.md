# 🥬 Analyseur de Têtê : Classification IA des Légumes

<p align="center">
  <a href="https://streamlit.io/">
    <img src="https://img.shields.io/badge/Streamlit-FF6B35?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  </a>
  <a href="https://python.org/">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  </a>
  <a href="https://tensorflow.org/">
    <img src="https://img.shields.io/badge/TensorFlow-FF6B35?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  </a>
</p>

---

## 📝 Description

L'**Analyseur de Têtê** est une application IA qui classe les images de Têtê en :

- ✅ Bonne qualité  
- ❌ Mauvaise qualité  
- ⚠️ Non-Têtê (détecte si ce n'est pas un Têtê)

**Techniques utilisées :**  
- MobileNetV2 pour l’extraction de caractéristiques  
- PCA pour la réduction dimensionnelle  
- XGBoost pour la classification optimisée  

**Interface web** : Streamlit pour uploader une photo ou utiliser la webcam.

**Objectif** : Évaluer rapidement la qualité des Têtês au marché ou en cuisine.

**Démonstration :** [Streamlit déployé](https://vegetablesimagesclassification-xdvkukq6hw3lsifp94pjt6.streamlit.app/)

---

## ⚡ Installation

1. **Cloner le dépôt**
```bash
git clone https://github.com/nuvnce/Vegetables_images_classification.git
cd Vegetables_images_classification
````

2. **Créer un environnement virtuel** (recommandé)

```bash
python -m venv env
# Linux / Mac
source env/bin/activate
# Windows
env\Scripts\activate
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Télécharger les données**

* [Google Drive](https://drive.google.com/drive/folders/1LX1JKdT2XbWArc-RtMJg_RsrdQs8PCM3?usp=sharing)
* Extraire dans `dataset/` :

  ```
  dataset/
  ├── bon/
  └── mauvais/
  ```

5. **Générer les modèles**

```bash
python main.py
```

* Crée :

  * `data_augmented/` → images augmentées
  * `model_out/` → features, PCA, modèles initiaux
  * `train_out/` → modèle final optimisé

---

## 🚀 Utilisation

1. **Pipeline complet**

```bash
python main.py
```

* Augmentation des données, extraction de features, comparaison et optimisation.

2. **Interface Streamlit**

```bash
streamlit run app.py
```

* Ouvrir : `http://localhost:8501`
* Charger une image ou prendre une photo
* Cliquer "Prédire" pour le verdict

**Exemples de sortie :**

* ✅ Bonne qualité : 92 % confiance
* ❌ Mauvaise qualité : 87 % confiance
* ⚠️ Non-Têtê : avertissement si < 70 % confiance

---

## 📂 Structure du projet

```
classification/
├── dataset/            # Images originales (télécharger depuis Drive)
│   ├── bon/
│   └── mauvais/
├── data_augmented/     # Images augmentées générées par main.py
├── model_out/          # Features, PCA, modèles initiaux
├── train_out/          # Modèle optimisé final
├── app.py              # Interface Streamlit
├── main.py             # Orchestration pipeline
├── module1.py          # Extraction & comparaison
├── module2.py          # Optimisation
├── module3.py          # Augmentation
├── requirements.txt    # Dépendances
└── README.md           # Ce fichier
```

---

## 📊 Résultats

* **Précision** : 90 % sur le jeu de test
* **AUC** : 0.90 (courbe ROC)
* **Graphiques** : Voir `model_out/` et `train_out/` (PCA, matrices de confusion, ROC)

📄 Rapport complet : [lien](https://drive.google.com/file/d/1ByggdRM5Eflu7oTJeRxax4x6sk-YwEUx/view?usp=sharing)

---

## 🛠 Dépendances

Voir `requirements.txt` :
TensorFlow, NumPy, Matplotlib, Scikit-learn, XGBoost, Streamlit, etc.

---

## 📝 Licence

MIT License – Utilisez librement, citez-moi si possible !

---

## 👥 Contributeurs

* **Daniel ESSONANI** – Développement principal

---

## 💬 Support

Pour tout problème ou question : créer une issue sur GitHub ou me contacter directement.

Amusez-vous bien avec vos Têtês ! 🥬

```

