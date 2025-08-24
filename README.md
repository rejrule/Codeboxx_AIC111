# AIC-111 – Video Reconstruction

## 1) Objectif
Reconstituer des vidéos d'ascenseur à partir de frames mélangées en :
- Extrayant date + numéro de frame via **Azure OCR**
- Classant/renommant les frames
- **Reconstruisant** des vidéos propres avec **OpenCV**
- Déployant un petit endpoint **AWS SAM** (PyTorch inference)

## 2) Arborescence
(voir structure du repo)

## 3) Installation
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env && nano .env                   # Renseigner les clés Azure


4) Pipeline local

OCR → python -m src.ocr_extraction

Tri/Rename → python -m src.sort_frames

Vidéos → python -m src.video_reconstruction

5) Déploiement AWS (SAM)

Voir aws/README_AWS_SAM.md.


Codeboxx_AIC111/
│
├── data/
│   ├── frames_raw/          # Images récupérées (désordonnées, noms aléatoires)
│   ├── frames_sorted/       # Images renommées et classées par date + frame_number
│   └── videos/              # Vidéos reconstruites (mod11_video1.avi, etc.)
│
├── src/
│   ├── ocr_extraction.py    # Extraction date + frame_number avec Azure OCR
│   ├── sort_frames.py       # Classement/renommage des frames
│   └── video_reconstruction.py # Reconstruction vidéo avec OpenCV
│
├── aws/
│   └── test-inference/      # Projet généré par `sam init` (API ML sur AWS)
│
├── notebooks/
│   └── AIC111_tests.ipynb   # Pour tester et documenter ton code étape par étape
│
├── deliverables/
│   └── deliverable_AIC111.txt   # Nom complet + liens vidéos + URL API
│
├── requirements.txt         # Dépendances Python (opencv-python, azure-ai-vision, etc.)
└── README.md                # Explications du projet (instructions rapides)




# 1) Définir les secrets Azure (exemples)
# PowerShell
$env:AZURE_CV_ENDPOINT=""
$env:AZURE_CV_KEY=""



# 2) Lancer le script
python src/ocr_extraction.py