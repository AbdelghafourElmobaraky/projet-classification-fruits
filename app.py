import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd # Pour afficher les tableaux de résultats
from PIL import Image

# 1. Configuration de la page
st.set_page_config(
    page_title="Classification de Fruits - EMSI", 
    page_icon="🍎",
    layout="wide" # Mode large pour mieux voir les tableaux
)

# En-tête
st.title("🍎🍌🍊 Classification de Fruits")
st.markdown(
    """
    **Réalisé par :** Abdelghafour Elmobaraky | **EMSI 2025-2026** - 5IIR G1  
    **Description :** Système de classification d'images (Pomme, Banane, Orange).
    """
)
st.markdown("---")

# 2. Chargement du modèle
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('fruit_model.h5')
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Erreur : Impossible de charger 'fruit_model.h5'. {e}")

class_names = ["apple", "banana", "orange"]
img_height, img_width = 32, 32

# Fonction utilitaire pour préparer l'image
def predict_image(image_file):
    # 1. Charger l'image et FORCER la conversion en RGB (Couleur 3 canaux)
    # Cela corrige le bug des images Noir & Blanc (1 canal) ou PNG transparents (4 canaux)
    img = Image.open(image_file).convert('RGB') 
    
    # 2. Redimensionner l'image
    img_resized = img.resize((img_width, img_height))
    
    # 3. Convertir en array
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0) # Créer un batch de 1 image

    # 4. Prédiction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    return img, predicted_class, confidence, score
    # Charger et redimensionner l'image
    img = Image.open(image_file)
    img_resized = img.resize((img_width, img_height))
    
    # Convertir en array
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0) # Créer un batch de 1 image

    # Prédiction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    return img, predicted_class, confidence, score

# ==========================================
# BARRE LATÉRALE : CHOIX DU MODE
# ==========================================
st.sidebar.header("Options")
mode = st.sidebar.radio(
    "Choisissez le mode de test :",
    ("Une seule image", "Plusieurs images (Dossier)")
)

# ==========================================
# MODE 1 : UNE SEULE IMAGE
# ==========================================
if mode == "Une seule image":
    st.subheader("🖼️ Test sur une seule image")
    
    uploaded_file = st.file_uploader("Chargez une image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption='Image source', width=300)
        
        with col2:
            st.write("Analyse en cours...")
            _, pred_class, conf, scores = predict_image(uploaded_file)
            
            # Affichage du résultat avec couleur
            if conf > 80:
                st.success(f"C'est une **{pred_class.upper()}** ({conf:.2f}%)")
            else:
                st.warning(f"C'est probablement une **{pred_class.upper()}** ({conf:.2f}%)")
                
            # Graphique de probabilité
            chart_data = pd.DataFrame(
                [scores.numpy()], 
                columns=["Apple", "Banana", "Orange"]
            )
            st.bar_chart(chart_data.T)

# ==========================================
# MODE 2 : PLUSIEURS IMAGES (Simulation Dossier)
# ==========================================
elif mode == "Plusieurs images (Dossier)":
    st.subheader("📂 Test sur un lot d'images")
    st.info("Sélectionnez toutes les images de votre dossier en une seule fois.")

    # accept_multiple_files=True permet de charger tout un "dossier"
    uploaded_files = st.file_uploader(
        "Chargez vos images (Ctrl+A pour tout sélectionner)...", 
        type=["jpg", "png", "jpeg"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"Analyse de {len(uploaded_files)} images...")
        
        results = []
        progress_bar = st.progress(0)
        
        # Boucle sur toutes les images
        for i, file in enumerate(uploaded_files):
            _, pred_class, conf, _ = predict_image(file)
            
            # Enregistrer les résultats
            results.append({
                "Nom du fichier": file.name,
                "Prédiction": pred_class.upper(),
                "Confiance (%)": f"{conf:.2f}"
            })
            # Mettre à jour la barre de progression
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        # Création d'un DataFrame (Tableau)
        df = pd.DataFrame(results)
        
        # Affichage du tableau
        st.dataframe(df, use_container_width=True)
        
        # Statistiques
        st.subheader("📊 Résumé Statistique")
        col1, col2 = st.columns(2)
        
        with col1:
            # Compter combien de pommes, bananes, oranges
            counts = df["Prédiction"].value_counts()
            st.bar_chart(counts)
            
        with col2:
            st.write("Répartition détectée :")
            st.write(counts)