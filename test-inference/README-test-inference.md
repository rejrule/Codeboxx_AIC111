# test-inference — Déploiement d’un modèle PyTorch avec AWS SAM (AIC111)

Ce dossier contient une **API serverless** qui reçoit un nombre `x` et retourne la prédiction `y` d’une fonction approximée (ici un petit modèle PyTorch équivalent à **y = 3x + 1**).  
L’API est packagée en **image Docker** et déployée sur **AWS Lambda** derrière **API Gateway**.

---

## ✅ URL publique (déjà déployée)

> **POST** `https://seyq4lqbog.execute-api.us-east-1.amazonaws.com/Prod/classify_digit/`  
> **Headers**: `Content-Type: application/json`  
> **Body**: `{"x": 2}`  
> **Réponse attendue**: `{"prediction": 7.0}`

### Tester rapidement (curl)
```bash
curl -s -X POST "https://seyq4lqbog.execute-api.us-east-1.amazonaws.com/Prod/classify_digit/" \
  -H "Content-Type: application/json" \
  -d '{"x": 2}'
# {"prediction": 7.0}
```

> ℹ️ **Notes**  
> - La première requête peut prendre quelques secondes (**cold start**). Si vous voyez `{"message": "Endpoint request timed out"}`, relancez la même requête et elle devrait réussir.  
> - Le **timeout API Gateway** est 30 s.

---

## 📁 Structure utile du projet

```
test-inference/
├─ app/
│  ├─ app.py               # Code de la Lambda (handler: app.lambda_handler)
│  └─ requirements.txt     # Dépendances (CPU-only) : torch==2.2.2
├─ events/
│  └─ ping.json            # Événement d’exemple pour sam local invoke
├─ template.yaml           # Définition SAM (Lambda + API Gateway)
└─ README.md               # Ce fichier (à placer ici)
```

- **app/app.py** : charge un mini-modèle PyTorch (ou valeurs par défaut) et retourne `{"prediction": <float>}`.  
- **app/requirements.txt** : dépendances minimalistes (PyTorch CPU) pour éviter les conflits.  
- **template.yaml** :  
  - Chemin API: **POST** `/classify_digit`  
  - **MemorySize**: 512 MB  
  - **PackageType**: Image (base `public.ecr.aws/lambda/python:3.9`)  
  - Pas d’authentification (API publique pour besoins du cours).  

---

## 🧪 Tests **locaux** (Docker + SAM)

### Prérequis
- **Docker Desktop** en marche
- **AWS SAM CLI**
- (facultatif) Python 3.x

### Lancer en local
```bash
cd test-inference

# Build de l’image Lambda via Docker
sam build --use-container

# Démarrer l’API locale (port 3000)
sam local start-api
```

Dans un autre terminal, appelez l’API locale :
```bash
curl -s -X POST http://127.0.0.1:3000/classify_digit \
  -H "Content-Type: application/json" \
  -d '{"x": 2}'
# {"prediction": 7.0}
```

### Invocation locale directe (événement API Gateway)
```bash
# Exemple d’event (déjà fourni dans events/ping.json):
# {
#   "resource": "/classify_digit",
#   "path": "/classify_digit",
#   "httpMethod": "POST",
#   "headers": { "Content-Type": "application/json" },
#   "isBase64Encoded": false,
#   "body": "{\"x\": 2}"
# }

sam local invoke InferenceFunction -e events/ping.json
# ... "body":"{\"prediction\": 7.0}"
```

---

## 🚀 (Re)déploiement sur AWS (propriétaire du compte)

> ⚠️ Non requis pour le coach qui **teste** seulement l’URL publique.  
> Ces commandes sont pour le propriétaire du compte AWS.

Prérequis :  
- **AWS CLI** configuré (`aws configure`) ou `AWS_PROFILE=sam-deployer`
- **Docker** + **SAM CLI**

```bash
cd test-inference

# Build image (Docker)
sam build --use-container

# Premier déploiement guidé (répondre aux invites)
sam deploy --guided
# Région: us-east-1
# Confirmer changements: Y
# Autoriser création de rôles IAM: Y
# Désactiver rollback: N
# Pas d’auth pour InferenceFunction: Y
# Sauver la config: Y

# Déploiements suivants
sam deploy
```

En fin de déploiement, SAM affiche les **Outputs**, notamment :  
- `InferenceApi` → l’URL publique (`.../Prod/classify_digit/`)  
- `InferenceFunction` → ARN de la Lambda  
- `InferenceFunctionIamRole` → Rôle IAM

---

## 🔍 Dépannage

- **Timeout à la première requête** : relancer la requête (cold start).  
- **Logs CloudWatch (déployé)** :
```bash
sam logs -n InferenceFunction --stack-name "test-inference" --tail
```
- **Tester localement sans API** :
```bash
sam local invoke InferenceFunction -e events/ping.json
```
- **Docker** doit être **running** pour `sam build --use-container` et `sam local start-api`.

---

## 🧹 Nettoyage (propriétaire du compte AWS)

Pour supprimer toutes les ressources AWS créées par cette stack :
```bash
sam delete --stack-name "test-inference"
```

---

## 📦 Détails techniques rapides

- **Runtime** : Python 3.9 (image `public.ecr.aws/lambda/python:3.9`)  
- **Inference** : PyTorch CPU (`torch==2.2.2`)  
- **Mémoire Lambda** : 512 MB  
- **Endpoint** : `POST /classify_digit`  
- **Corps attendu** : `{"x": <nombre>}`  
- **Réponse** : `{"prediction": <nombre>}`

---

## 👤 Livrables (pour le cours)

Inclure dans votre livrable :
- Votre **nom complet**  
- **Lien vers vos 5 vidéos** (format `mod11_video<n>.mp4` ou `.avi`)  
- **URL de l’API** : `https://seyq4lqbog.execute-api.us-east-1.amazonaws.com/Prod/classify_digit/`

---

Bon test !
