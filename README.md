# DEVANET
# Multimodal Sentiment Analysis – MOSI (Texte + Audio + Vidéo)


Ce projet implémente un modèle de **sentiment multimodal** capable d’analyser simultanément :

- 📝 **Texte**  
- 🎤 **Audio**  
- 🎥 **Vidéo (frames)**  
- 😶 **Features OpenFace (AUs / Vision)**  
- 🗣️ **Descriptions textuelles des modalités (D_a, D_v,D_ann)**  
- 📌 **Annotations humaines**

Le modèle fusionne ces modalités via un réseau **Fully Connected Multimodal (DevaModel)** et prédit un score de sentiment continu.

---

## 🚀 Fonctionnalités Implémentées

### 🔹 1. **Encodage du Texte**
- BERT (HuggingFace) pour encoder :  
  - transcription (`X_t`)  
  - description audio (`D_a`)  
  - description vidéo (`D_v`)  
  - annotation humaine (`D_ann`)

### 🔹 2. **Encodage Audio**
- Chargement des signaux audio (.npy)  
- Extraction MFCC / embeddings (selon preprocessing)

### 🔹 3. **Encodage Vidéo**
- Chargement des frames vidéo par segment  
- Chargement des AUs (vision features) depuis `mosi_data.pkl`

### 🔹 4. **Fusion Multimodale**
Modèle personnalisé :
- Projection de toutes les modalités  
- Concaténation  
- MLP final → prédiction du score de sentiment

### 🔹 5. **Deux versions de modèles Enregistrés**
- `deva_model_30_non_normaliser_Emotional_Embedding.pth`  (version finale)
- `deva_model_30_epoch_normaliser_Bert_Embedding.pth` 

---

## 📦 Données Utilisées

Les données viennent du dataset **CMU-MOSI**, un benchmark multimodal pour le sentiment.

### 📁 Sources utilisées dans ce projet :
- `CMU-MOSI/label.csv` → textes, annotations, labels  
- `mosi_data.pkl` → vision features (AUs) + IDs  
- `train_mosi_frames_list.npy` → frames vidéo (train)  
- `train_mosi_audio_list.npy` → audios (train)  
- `test_mosi_frames_list.npy` → frames vidéo (test)  
- `test_mosi_audio_list.npy` → audios (test)

---

## 🔗 Lien vers la Data

Toutes les données utilisées dans ce projet sont accessibles ici :

👉 **Google Drive (Données + Modèles)**  
https://drive.google.com/drive/folders/1StfuzhPdk-6V1JLLF7u0C0HhQr25ZqwH

---

## 🧠 Organisation du Dataset

| Modalité | Format | Description |
|----------|--------|-------------|
| Texte | `label.csv` | Transcriptions + annotations + score (-3 → +3) |
| Audio | `.npy` | Liste des signaux ou MFCC |
| Vidéo (frames) | `.npy` | Images extraites de chaque segment |
| Vision / AUs | `mosi_data.pkl` | OpenFace features |
| IDs MOSI | `mosi_data.pkl` | Identifiants de segments |

---
# 🔥 Fusion Multimodale Implémentée

Le cœur du projet repose sur une **fusion multimodale hiérarchique**, composée de :

### 🔹 1. **Projection indépendante**
Chaque modalité est d’abord projetée dans un espace commun :

- BERT → vectorisation du texte, D_a, D_v, D_ann  
- Audio MFCC → vecteur 20D  
- Vision → vecteur OpenFace  
- Frames vidéo → embeddings pré-extraits  

Chaque vecteur passe dans une couche linéaire :  


---

## 🧪 Entraînement & Tests

- Entraînement effectué pendant **30 époques**
- Version finale :  
  `deva_model_30_epoch_normaliser_Bert_Embedding.pth`
- Prédiction d’un score continu de sentiment

---

## 📜 Licence
Projet académique — ECE Paris.

---


