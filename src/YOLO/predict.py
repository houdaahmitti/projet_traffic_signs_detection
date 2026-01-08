from ultralytics import YOLO
import os

# 🔹 Définir le projet et les chemins
PROJECT_DIR = "/content/drive/MyDrive/projet/projet_traffic_signs_detection"
MODEL_PATH = os.path.join(PROJECT_DIR, "runs/train/train_signs43/weights/best.pt")
IMAGE_PATH = os.path.join(PROJECT_DIR, "data/images/val/00500.jpg")

# 🔹 Charger le modèle YOLOv8 entraîné
print("🔍 Loading trained model...")
model = YOLO(MODEL_PATH)

# 🔹 Faire la prédiction sur l'image
print("📸 Running prediction on image...")
results = model.predict(
    source=IMAGE_PATH,  # image à tester
    imgsz=768,          # taille de l'image pour le modèle
    conf=0.25,          # seuil de confiance minimum
    project=os.path.join(PROJECT_DIR, "runs/detect"),  # dossier de sauvegarde des résultats
    name="test_sign",       # sous-dossier pour cette prédiction
    save=True               # sauvegarder l'image avec les boîtes détectées
)

print("✅ Prediction finished!")

# 🔹 Afficher le résultat dans Colab
results[0].plot()  # optionnel pour visualiser directement
