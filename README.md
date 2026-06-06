# Système de Recommandation Musicale Spotify
Bienvenue dans le dépôt du **Système de Recommandation Musicale Spotify** ! 🎵  
Ce projet présente une application interactive de machine learning permettant de recommander des chansons similaires à partir des préférences d'un utilisateur. Conçu comme projet de portfolio, il met en avant les bonnes pratiques en data engineering et développement Python : pipeline de clustering, architecture modulaire, clean code et documentation structurée.

---

## 🎯 Contexte

Le dataset provient de **Spotify** et couvre plus de **170 000 chansons** s'étalant de **1921 à 2020**, enrichies de leurs features audio (énergie, dansabilité, acousticité, tempo…). L'application exploite ces caractéristiques pour recommander des chansons par similarité, sans historique d'écoute ni profil utilisateur.

---

## 🚀 Exigences du projet

### Pipeline de données (Data Engineering)

#### Objectif
Construire un pipeline de traitement structuré depuis l'exploration des données brutes jusqu'à l'alimentation du système de recommandation.

#### Spécifications
- **Sources de données** : Quatre fichiers CSV Spotify (`data.csv`, `data_by_genres.csv`, `data_by_year.csv`, `data_by_artist.csv`).
- **Analyse exploratoire** : Distribution temporelle, évolution des features audio, top genres et artistes — documentée dans `notebooks/EDA.ipynb`.
- **Clustering** : Regroupement des genres (KMeans, 12 clusters) et des chansons (KMeans, 25 clusters) avec visualisation t-SNE et PCA — documenté dans `notebooks/Clustering.ipynb`.
- **Modélisation** : Construction et validation du système de recommandation par similarité cosinus — documenté dans `notebooks/Modelisation.ipynb`.
- **Documentation** : Convention de nommage disponible dans `docs/naming_conventions.md`.

---

### Recommandation & Interface (Machine Learning + Streamlit)

#### Objectif
Fournir une interface interactive multi-pages permettant la recommandation en temps réel, l'exploration des données et la visualisation des clusters.

#### Fonctionnalités
- **Recommandation** : Saisie d'une chanson (nom + année) → 10 recommandations similaires par distance cosinus dans l'espace audio standardisé. Les chansons absentes du dataset sont récupérées via l'**API Spotify**.
- **EDA** : Histogrammes, courbes temporelles, bar charts et nuages de mots interactifs.
- **Clustering** : Projection PCA 2D des 170 000 chansons colorées par cluster.

Pour plus de détails sur les dépendances, consultez [requirements.txt](requirements.txt).

---

## 🗂️ Structure du projet

```
spotify-recommender/
│
├── app.py                          # Point d'entrée — streamlit run app.py
│
├── src/
│   ├── data_loader.py              # Chargement des fichiers CSV
│   ├── model.py                    # Pipeline StandardScaler + KMeans
│   └── recommender.py              # Logique de recommandation par similarité cosinus
│
├── data/
│   ├── data.csv                    # Dataset principal (170k+ chansons)
│   ├── data_by_genres.csv          # Agrégation par genre
│   ├── data_by_year.csv            # Agrégation par année
│   ├── data_by_artist.csv          # Agrégation par artiste
│   └── data_w_genres.csv           # Dataset enrichi avec genres
│
├── notebooks/
│   ├── Lecture_des_données.ipynb   # Exploration initiale des sources
│   ├── EDA.ipynb                   # Analyse exploratoire des données
│   ├── Clustering.ipynb            # Expérimentation KMeans et t-SNE
│   └── Modelisation.ipynb          # Construction du système de recommandation
│
├── docs/
│   └── naming_conventions.md       # Conventions de nommage du projet
│
├── .streamlit/
│   └── secrets.toml                # Clés API Spotify (non versionné)
│
├── requirements.txt                # Dépendances du projet
└── README.md                       # Présentation du projet
```

---

## ⚙️ Installation et utilisation

### 1. Cloner le dépôt
```bash
git clone https://github.com/Majin-M/spotify-recommender.git
cd spotify-recommender
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Configurer les clés API Spotify
Créer le fichier `.streamlit/secrets.toml` :
```toml
[spotify]
client_id     = "votre_client_id"
client_secret = "votre_client_secret"
```
> Les clés sont disponibles sur le [Spotify Developer Dashboard](https://developer.spotify.com/dashboard).

### 4. Lancer l'application
```bash
streamlit run app.py
```

---

## 🛡️ Licence
Ce projet est sous licence [MIT](LICENSE). Vous êtes libre de l'utiliser, le modifier et le partager avec attribution appropriée.

## 👤 À propos de moi
Je suis Steven Mouthoud, Data Engineer passionné par la construction de pipelines de données robustes et le développement d'applications orientées data.
