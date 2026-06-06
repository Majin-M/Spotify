# ==============================================================================
# Module : Entraînement du pipeline de clustering
# ==============================================================================
# Objectif du module :
#     Ce module contient la logique d'entraînement du pipeline KMeans
#     utilisé pour regrouper les chansons par similarité audio.
#
#     Il effectue les actions suivantes :
#     - Sélectionne les colonnes numériques du dataset.
#     - Standardise les features via StandardScaler.
#     - Regroupe les chansons en clusters via KMeans.
#
# Exemple d'utilisation :
#     from src.model import train_song_pipeline
# ==============================================================================


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ==============================================================================
# Constantes
# ==============================================================================

SONG_PIPELINE_PARAMS = {
    "n_clusters":  25,
    "random_state": 42,
    "verbose":     False
}


# ==============================================================================
# Entraînement du pipeline
# ==============================================================================

def train_song_pipeline(df: pd.DataFrame) -> Pipeline:
    """
    Entraîne un pipeline StandardScaler + KMeans sur les features numériques
    du dataset de chansons.

    Paramètres :
        df : DataFrame principal des chansons

    Retourne :
        Pipeline scikit-learn entraîné (scaler + kmeans)
    """
    X = _select_numeric_features(df)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(**SONG_PIPELINE_PARAMS))
    ], verbose=False)

    pipeline.fit(X)
    return pipeline


# ==============================================================================
# Fonctions internes
# ==============================================================================

def _select_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sélectionne uniquement les colonnes numériques du DataFrame.

    Paramètres :
        df : DataFrame source

    Retourne :
        DataFrame contenant uniquement les colonnes numériques
    """
    return df.select_dtypes(np.number)