Voici le contenu nettoyé et formaté correctement. Tu n'as plus qu'à copier le bloc de code ci-dessous et le coller dans ton fichier **`README.md`**
# 🍎 Classification d'Images de Fruits (CNN)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **Projet de Deep Learning** réalisé dans le cadre du module d'Intelligence Artificielle.
> Le but est de classifier des images de fruits (Pomme, Banane, Orange) en utilisant un réseau de neurones convolutif (CNN).


## 👤 Auteur
* **Étudiant :** Abdelghafour Elmobaraky
* **École :** EMSI (École Marocaine des Sciences de l'Ingénieur)
* **Année :** 2025-2026
* **Classe :** 5IIR G1


## 📋 Fonctionnalités de l'Application

L'application Web, développée avec **Streamlit**, offre deux modes d'utilisation :

### 1. 🖼️ Mode Image Unique
* Upload d'une seule image (jpg, png, jpeg).
* Prédiction instantanée de la classe (Apple, Banana, Orange).
* Affichage du score de confiance (%).
* Graphique des probabilités pour les 3 classes.

### 2. 📂 Mode Batch (Dossier)
* Upload simultané de plusieurs images.
* Traitement en boucle et génération d'un tableau de données (Pandas DataFrame).
* Statistiques globales sur le lot d'images (ex: "Combien de pommes détectées ?").


## 🛠️ Technologies Utilisées
* **TensorFlow / Keras :** Construction et entraînement du modèle CNN.
* **Streamlit :** Création de l'interface utilisateur interactive.
* **Pandas & NumPy :** Manipulation des données et résultats.
* **Pillow (PIL) :** Traitement d'images (Redimensionnement & Conversion).

## ⚙️ Structure du Projet

PROJET/
│
├── dataset/                  # Dossier contenant les images (Train/Val/Test)
├── app.py                    # Application Streamlit (Interface Web)
├── train_model.py            # Script d'entraînement du modèle
├── requirements.txt          # Liste des dépendances
├── fruit_model.h5            # Le modèle entraîné (généré par train_model.py)
└── README.md                 # Documentation du projet

## 🚀 Installation et Lancement (Local)

Suivez ces étapes pour tester le projet sur votre machine :

### 1\. Cloner ou télécharger le projet

Placez-vous dans le dossier du projet via le terminal.

### 2\. Installer les dépendances

pip install -r requirements.txt

### 3\. Entraîner le modèle (Si 'fruit\_model.h5' n'existe pas)

Ce script va lire le dataset, entraîner le CNN et sauvegarder le fichier `.h5`.

python train_model.py

### 4\. Lancer l'application

streamlit run app.py

Une page web s'ouvrira automatiquement dans votre navigateur.

## 💡 Choix Techniques et Résolution de Problèmes

### Gestion des Canaux d'Image (Bug Fix)

Durant le développement, une erreur `ValueError` survenait lors de l'upload d'images en niveaux de gris (1 canal) ou PNG (4 canaux), car le modèle attendait strictement du RGB (3 canaux).

**Solution implémentée dans `app.py` :**
Nous forçons la conversion de chaque image entrée en RGB avant le traitement :

# Force l'image en 3 canaux (R, G, B) pour éviter les erreurs de dimension
img = Image.open(image_file).convert('RGB')

### Architecture du Modèle

Le modèle est un CNN séquentiel classique comprenant :

1.  **Rescaling :** Normalisation des pixels (0-1).
2.  **Conv2D & MaxPooling :** 3 blocs pour l'extraction de caractéristiques.
3.  **Dense Layers :** Classification finale (Softmax).

## 🌐 Déploiement

Ce projet est configuré pour être déployé gratuitement sur **Streamlit Community Cloud**.

1.  Pousser le code sur GitHub.
2.  Connecter le repository à Streamlit Cloud.
3.  Déployer \!
