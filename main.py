import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# 1. Initialisation de l'application FastAPI
app = FastAPI(
    title="Fruit Classification API",
    description="API Backend pour l'application mobile de détection de fruits (EMSI)",
    version="1.0"
)

# 2. Chargement du modèle (Une seule fois au démarrage)
# On le met en variable globale pour ne pas le recharger à chaque requête
print("Chargement du modèle 'fruit_model.h5'...")
try:
    model = tf.keras.models.load_model('fruit_model.h5')
    print("Modèle chargé avec succès !")
except Exception as e:
    print(f"ERREUR CRITIQUE : Impossible de charger le modèle. {e}")
    model = None

# Les classes doivent être dans le même ordre que l'entraînement
class_names = ["apple", "banana", "orange"]

# 3. Route de test (Health Check)
@app.get("/")
def index():
    return {"status": "online", "message": "Bienvenue sur l'API de classification de fruits"}

# 4. Route de Prédiction (C'est celle-ci que Flutter va appeler)
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Prend une image en entrée, la traite et retourne la prédiction.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Le modèle n'est pas chargé sur le serveur.")

    try:
        # A. Lire le fichier image envoyé par le mobile
        contents = await file.read()
        
        # B. Ouvrir l'image et forcer la conversion en RGB
        # (Indispensable pour éviter l'erreur des 4 canaux ou du Noir & Blanc)
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # C. Prétraitement (Même logique que lors de l'entraînement)
        # Redimensionner en 32x32 pixels
        image = image.resize((32, 32))
        
        # Convertir en tableau numpy
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        
        # Créer un batch de 1 image (dimension : 1, 32, 32, 3)
        img_array = tf.expand_dims(img_array, 0)

        # D. Prédiction
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        # Trouver la classe avec le score le plus haut
        predicted_class = class_names[np.argmax(score)]
        confidence = float(np.max(score) * 100)

        # E. Retourner la réponse en JSON
        return {
            "filename": file.filename,
            "prediction": predicted_class,          # ex: "apple"
            "confidence": f"{confidence:.2f}%",     # ex: "98.50%"
            "raw_scores": {
                "apple": float(score[0]),
                "banana": float(score[1]),
                "orange": float(score[2])
            }
        }

    except Exception as e:
        return {"error": str(e)}

# Bloc optionnel pour lancer via "python main.py" directement
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)