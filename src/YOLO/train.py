import os
import torch
from ultralytics import YOLO

# Autoriser les duplications de librairie Intel MKL
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2️⃣ Définir les chemins
PROJECT_DIR = "/content/drive/MyDrive/projet/projet_traffic_signs_detection"
DATA_YAML = os.path.join(PROJECT_DIR, "data.yaml")  # data.yaml pour 43 classes

# Créer dossier pour sauvegarder les runs
os.makedirs(os.path.join(PROJECT_DIR, "runs", "train"), exist_ok=True)

# 3️⃣ Vérifier le GPU
device = "0" if torch.cuda.is_available() else "cpu"
print(f"⚡ Training will run on device: {device}")
if device != "cpu":
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")

# 4️⃣ Charger le modèle YOLOv8 pré-entraîné (m = medium)
print("🚀 Loading YOLOv8m model...")
model = YOLO("yolov8m.pt")  # tu peux aussi utiliser yolov8n.pt (nano) pour test rapide

# 5️⃣ Paramètres d'entraînement
TRAIN_PARAMS = {
    "data": DATA_YAML,       # chemin vers data.yaml
    "epochs": 50,            # nombre d'epochs
    "imgsz": 640,            # taille des images, 640 est un bon compromis
    "batch": 16 if device != "cpu" else 4,
    "workers": 2,
    "device": device,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "augment": True,
    "degrees": 10,
    "scale": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "project": os.path.join(PROJECT_DIR, "runs", "train"),
    "name": "train_signs43",
    "exist_ok": True
}

# 6️⃣ Lancer l'entraînement
print("⚡ Starting training...")
model.train(**TRAIN_PARAMS)
print("✅ Training completed!")

# 7️⃣ Optionnel: sauvegarde du modèle final
final_model_path = os.path.join(PROJECT_DIR, "runs", "train", "train_signs43", "weights", "best.pt")
print(f"Trained model saved at: {final_model_path}")
