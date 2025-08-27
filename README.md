# 🚀 AIC-111 – Video Reconstruction

## 📌 Objectif
Reconstituer des vidéos d'ascenseur à partir de frames mélangées en :
- 📖 **Extraction** de la date + numéro de frame via **Azure OCR**
- 🗂️ **Classement / renommage** des frames
- 🎥 **Reconstruction vidéo** avec **OpenCV**
- ☁️ Déploiement d’un petit endpoint **AWS SAM** (PyTorch inference)

---

## 📂 Arborescence du projet
```
Codeboxx_AIC111/
│
├── data/
│   ├── frames_raw/          # Images brutes (désordonnées, noms aléatoires)
│   ├── frames_sorted/       # Images renommées et classées par date + frame_number
│   └── videos/              # Vidéos reconstruites (mod11_video1.avi, etc.)
│
├── src/
│   ├── ocr_extraction.py        # Extraction date + frame_number avec Azure OCR
│   ├── sort_frames.py           # Classement/renommage des frames
│   └── video_reconstruction.py  # Reconstruction vidéo avec OpenCV
│
├── aws/
│   └── test-inference/      # Projet généré par `sam init` (API ML sur AWS)
│
├── notebooks/
│   └── AIC111_tests.ipynb   # Tests et documentation étape par étape
│
├── deliverables/
│   └── deliverable_AIC111.txt   # Nom complet + liens vidéos + URL API
│
├── requirements.txt         # Dépendances Python (opencv-python, azure-ai-vision, etc.)
└── README.md                # Explications et guide rapide
```

---

## ⚙️ Installation

### 1. Créer un environnement virtuel
```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\activate
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Configurer les clés Azure
Créer un fichier `.env` (copie de `.env.example`) ou exporter directement :

#### PowerShell (Windows)
```powershell
$env:AZURE_CV_ENDPOINT="https://xxxx.cognitiveservices.azure.com/"
$env:AZURE_CV_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

#### Bash (Linux/Mac)
```bash
export AZURE_CV_ENDPOINT="https://xxxx.cognitiveservices.azure.com/"
export AZURE_CV_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

---

## 🔎 Pipeline local

### 1. Extraction OCR (date + frame)
```bash
# Exécuter sur toutes les images
python src/ocr_extraction.py

# Tester sur 20 images seulement (éviter quota Azure)
export MAX_IMAGES=20
export OCR_SLEEP=0.40
export VERBOSE=1
export OCR_DEBUG=0

python src/ocr_extraction.py
```

Résultat attendu :
- `data/frames_metadata.csv` (filename, date, frame_number)
- `data/frames_sorted/` (images triées par date)
- Logs détaillés en console

---

### 2. Tri & renommage des frames
```bash
python src/sort_frames.py
```

---

### 3. Reconstruction vidéo
Plusieurs modes sont possibles :

```bash
# 1) Vidéo unique, ordre CSV, avec overlay date/frame
python src/video_reconstruction.py --use-csv --fps 15 --overlay

# 2) Un MP4 par date (nécessite CSV + frames_sorted)
python src/video_reconstruction.py --use-csv --split-by-date --fps 15 --overlay

# 3) Rapide sans CSV, prend tout ce qu’il trouve
python src/video_reconstruction.py --fps 15

# 4) Tester sur 20 images seulement
python src/video_reconstruction.py --use-csv --fps 15 --limit 20 --overlay
```

---

## ☁️ Déploiement AWS (SAM)
Un projet AWS SAM est inclus dans `aws/test-inference/`.

➡️ Voir [aws/README_AWS_SAM.md](aws/README_AWS_SAM.md) pour :
- Créer un container
- Déployer la Lambda d’inférence
- Tester via API Gateway

---

## ✅ Résultats attendus
- OCR robuste → CSV + JSON avec **date et numéro de frame**
- Images correctement triées → `frames_sorted/`
- Vidéos reconstruites → `data/videos/*.mp4`
- Déploiement optionnel sur AWS pour démo API

---

## 🛠️ Commandes utiles

### Activer/désactiver l’environnement
```bash
# Activer
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# Désactiver
deactivate
```

### Repartir à zéro
```bash
rm -rf data/frames_sorted data/videos data/ocr_debug
```
---

## 🧾 Auteur
Projet réalisé dans le cadre de **CodeBoxx – Module AIC-111**  
👉 Reconstituer des vidéos d’ascenseur par IA et OCR.
