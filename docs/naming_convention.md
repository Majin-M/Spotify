# **Conventions de Nommage**

Ce document décrit les conventions de nommage utilisées pour les modules, fonctions, variables, constantes et autres objets dans le projet de recommandation musicale Spotify.

## **Table des matières**

1. [Principes généraux](#principes-généraux)
2. [Conventions de nommage des fichiers](#conventions-de-nommage-des-fichiers)
3. [Conventions de nommage des fonctions](#conventions-de-nommage-des-fonctions)
   - [Règles de chargement](#règles-de-chargement)
   - [Règles d'entraînement](#règles-dentraînement)
   - [Règles de recommandation](#règles-de-recommandation)
   - [Règles internes](#règles-internes)
   - [Règles d'interface](#règles-dinterface)
4. [Conventions de nommage des variables](#conventions-de-nommage-des-variables)
   - [Constantes](#constantes)
   - [Variables de données](#variables-de-données)
   - [Variables de modèle](#variables-de-modèle)
5. [Glossaire des préfixes de fonctions](#glossaire-des-préfixes-de-fonctions)

---

## **Principes généraux**

- **Conventions de nommage** : Utiliser le `snake_case` pour les fonctions, variables et modules ; le `SCREAMING_SNAKE_CASE` pour les constantes.
- **Langue** : Utiliser l'anglais pour tous les noms d'objets Python.
- **Éviter les abréviations ambiguës** : Préférer des noms explicites et descriptifs à des raccourcis cryptiques.
- **Conformité PEP 8** : Respecter les conventions officielles de style Python.
- **Fonctions internes** : Préfixer avec `_` les fonctions non destinées à être appelées depuis l'extérieur du module.

---

## **Conventions de nommage des fichiers**

| Fichier | Rôle |
|---|---|
| `app.py` | Point d'entrée — interface Streamlit et orchestration des pages |
| `src/data_loader.py` | Chargement des fichiers CSV et création de la colonne `decade` |
| `src/model.py` | Entraînement du pipeline StandardScaler + KMeans |
| `src/recommender.py` | Logique de recommandation par similarité cosinus |
| `notebooks/Lecture_des_données.ipynb` | Exploration initiale des sources de données |
| `notebooks/EDA.ipynb` | Analyse exploratoire — visualisations et conclusions |
| `notebooks/Clustering.ipynb` | Expérimentation du clustering KMeans et t-SNE |
| `notebooks/Modelisation.ipynb` | Construction et validation du système de recommandation |

---

## **Conventions de nommage des fonctions**

### **Règles de chargement**
- Les fonctions de chargement d'une ressource doivent commencer par le préfixe `load_`.
- **`load_<ressource>`**
  - Exemple : `load_data` → Charge les quatre fichiers CSV du dataset Spotify.
  - Exemple : `load_resources` → Charge les données et le pipeline (cache Streamlit).

### **Règles d'entraînement**
- Les fonctions d'entraînement de pipeline ou de modèle doivent commencer par `train_`.
- **`train_<cible>`**
  - Exemple : `train_song_pipeline` → Entraîne le pipeline KMeans sur les chansons.

### **Règles de recommandation**
- Les fonctions de recommandation doivent utiliser un verbe d'action explicite.
- **`recommend_<cible>`**
  - Exemple : `recommend_songs` → Retourne les n chansons les plus similaires.

### **Règles internes**
- Les fonctions internes à un module (non exposées) doivent commencer par `_`.
- **`_<verbe>_<objet>`**
  - Exemple : `_get_song_data` → Récupère les données d'une chanson (local ou API).
  - Exemple : `_find_song_on_spotify` → Interroge l'API Spotify pour les features audio.
  - Exemple : `_get_mean_vector` → Calcule le vecteur moyen des chansons d'entrée.
  - Exemple : `_flatten_dict_list` → Transforme une liste de dicts en dict de listes.
  - Exemple : `_add_decade_column` → Ajoute la colonne `decade` au DataFrame.
  - Exemple : `_select_numeric_features` → Filtre les colonnes numériques du DataFrame.

### **Règles d'interface**
- Les fonctions de rendu Streamlit doivent commencer par le préfixe `render_`.
- **`render_<page>`**
  - Exemple : `render_recommendation_page` → Affiche la page de recommandation.
  - Exemple : `render_eda_page` → Affiche la page d'analyse exploratoire.
  - Exemple : `render_clustering_page` → Affiche la page de visualisation des clusters.

---

## **Conventions de nommage des variables**

### **Constantes**

| Constante | Module | Description |
|---|---|---|
| `DATA_PATH` | `data_loader.py` | Chemin vers le CSV principal des chansons |
| `GENRE_DATA_PATH` | `data_loader.py` | Chemin vers le CSV agrégé par genre |
| `YEAR_DATA_PATH` | `data_loader.py` | Chemin vers le CSV agrégé par année |
| `ARTIST_DATA_PATH` | `data_loader.py` | Chemin vers le CSV agrégé par artiste |
| `SONG_PIPELINE_PARAMS` | `model.py` | Hyperparamètres du pipeline KMeans |
| `NUMBER_COLS` | `recommender.py` | Colonnes numériques audio utilisées pour la recommandation |
| `METADATA_COLS` | `recommender.py` | Colonnes de métadonnées retournées dans les recommandations |
| `SOUND_FEATURES` | `app.py` | Features audio affichées dans les graphiques EDA |
| `PAGES` | `app.py` | Liste des pages de navigation Streamlit |

### **Variables de données**

| Nom | Type | Description |
|---|---|---|
| `data` | `pd.DataFrame` | DataFrame principal des chansons |
| `genre_data` | `pd.DataFrame` | DataFrame agrégé par genre |
| `year_data` | `pd.DataFrame` | DataFrame agrégé par année |
| `artist_data` | `pd.DataFrame` | DataFrame agrégé par artiste |
| `song_list` | `list` | Liste de dicts `{'name': str, 'year': int}` fournie en entrée |
| `song_center` | `np.ndarray` | Vecteur moyen des features audio des chansons d'entrée |
| `song_vectors` | `list` | Liste des vecteurs de features audio de chaque chanson |
| `scaled_data` | `np.ndarray` | Dataset standardisé pour le calcul des distances |
| `distances` | `np.ndarray` | Distances cosinus entre le vecteur centre et le dataset |
| `projection` | `pd.DataFrame` | Coordonnées 2D issues de la projection PCA pour la visualisation |

### **Variables de modèle**

| Nom | Type | Description |
|---|---|---|
| `pipeline` | `sklearn.Pipeline` | Pipeline entraîné (StandardScaler + KMeans) |
| `scaler` | `StandardScaler` | Étape de normalisation extraite du pipeline |
| `sp` | `spotipy.Spotify` | Client Spotify authentifié |

---

## **Glossaire des préfixes de fonctions**

| Préfixe | Signification | Exemple(s) |
|---|---|---|
| `load_` | Chargement de données ou de ressources | `load_data`, `load_resources` |
| `train_` | Entraînement d'un pipeline ou modèle ML | `train_song_pipeline` |
| `recommend_` | Génération de recommandations | `recommend_songs` |
| `render_` | Rendu d'une page ou composant Streamlit | `render_recommendation_page`, `render_eda_page` |
| `connect_` | Connexion à un service externe | `connect_spotify` |
| `_` (préfixe) | Fonction interne au module | `_find_song_on_spotify`, `_get_mean_vector` |