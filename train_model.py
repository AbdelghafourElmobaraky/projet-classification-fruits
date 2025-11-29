# Lab : CNN Image classification.
# Realisé par : Abdelghafour Elmobaraky EMSI 2025-2026 5IIR G1
# Reference Dataset : https://bitbucket.org/ishaanjav/code-and-deploy-custom-tensorflow-lite-model/raw/a4febbfee178324b2083e322cdead7465d6fdf95/fruits.zip

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Step 0 : Configuration ---
# Assurez-vous que le dossier "dataset/fruits" existe et contient train/validation/test
# Si vous êtes sur Colab/Jupyter, vous devez dézipper le fichier avant.

print("Version TensorFlow : ", tf.__version__)

# --- Step 1 : Dataset ---
img_height, img_width = 32, 32
batch_size = 20

# Vérification basique pour éviter les erreurs de chemin
if not os.path.exists("../lab1/dataset/fruits/train"):
    print("ERREUR: Le dossier 'dataset/fruits/train' est introuvable.")
    print("Veuillez vous assurer d'avoir extrait le fichier fruits.zip dans le dossier du projet.")
    exit()

print("Chargement des données d'entraînement...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    "../lab1/dataset/fruits/train",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

print("Chargement des données de validation...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    "../lab1/dataset/fruits/validation",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

print("Chargement des données de test...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    "../lab1/dataset/fruits/test",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Data Visualization (S'affichera dans une fenêtre pop-up)
class_names = ["apple", "banana", "orange"]
print(f"Classes détectées : {class_names}")

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
print("Fermez la fenêtre du graphique pour continuer l'entraînement...")
plt.show()

# --- Step 2 : Model ---
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)), # Normalization
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(3) # 3 Classes (Apple, Banana, Orange)
])

model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

# --- Step 3 : Train ---
print("Début de l'entraînement...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)

# --- Step 4 : Test ---
print("\nÉvaluation du modèle sur les données de test :")
loss, accuracy = model.evaluate(test_ds)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Visualisation des prédictions
plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
    classifications = model.predict(images)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        index = np.argmax(classifications[i])
        plt.title(f"Pred: {class_names[index]} | Real: {class_names[labels[i]]}")
        plt.axis("off")
print("Fermez la fenêtre du graphique pour finaliser la sauvegarde...")
plt.show()

# --- Step 5 : Deployment Export ---

# 1. Sauvegarde pour Streamlit (Format Keras .h5)
model.save('fruit_model.h5')
print("\n[SUCCÈS] Modèle sauvegardé sous 'fruit_model.h5' (Pour Streamlit)")

# 2. Sauvegarde pour Mobile (Format TFLite - Comme demandé dans le Lab original)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", 'wb') as f:
    f.write(tflite_model)
print("[SUCCÈS] Modèle sauvegardé sous 'model.tflite' (Pour Android/iOS)")