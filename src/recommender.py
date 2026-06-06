# ==============================================================================
# Module : Logique de recommandation musicale
# ==============================================================================
# Objectif du module :
#     Ce module contient la logique de recommandation de chansons
#     basée sur la similarité cosinus dans l'espace des features audio.
#
#     Il effectue les actions suivantes :
#     - Recherche les données d'une chanson dans le dataset local.
#     - Si absente, interroge l'API Spotify pour récupérer ses features audio.
#     - Calcule le vecteur moyen des chansons fournies en entrée.
#     - Identifie les chansons les plus proches par distance cosinus.
#
# Exemple d'utilisation :
#     from src.recommender import recommend_songs
# ==============================================================================


import numpy as np
import pandas as pd
import spotipy
import streamlit as st

from scipy.spatial.distance import cdist
from collections import defaultdict
from sklearn.pipeline import Pipeline


# ==============================================================================
# Constantes
# ==============================================================================

NUMBER_COLS = [
    "valence", "year", "acousticness", "danceability", "duration_ms",
    "energy", "explicit", "instrumentalness", "key", "liveness",
    "loudness", "mode", "popularity", "speechiness", "tempo"
]

METADATA_COLS = ["name", "year", "artists"]


# ==============================================================================
# Fonction principale
# ==============================================================================

def recommend_songs(
    song_list: list,
    spotify_data: pd.DataFrame,
    pipeline: Pipeline,
    sp: spotipy.Spotify,
    n_songs: int = 10
) -> list:
    """
    Recommande les n chansons les plus similaires à partir d'une liste de chansons.

    La similarité est calculée par distance cosinus dans l'espace
    des features audio standardisées.

    Paramètres :
        song_list     : Liste de dictionnaires {'name': str, 'year': int}
        spotify_data  : DataFrame principal des chansons
        pipeline      : Pipeline entraîné (scaler + kmeans)
        sp            : Client Spotify authentifié
        n_songs       : Nombre de recommandations à retourner (défaut : 10)

    Retourne :
        Liste de dictionnaires contenant les métadonnées des chansons recommandées
    """
    song_dict   = _flatten_dict_list(song_list)
    song_center = _get_mean_vector(song_list, spotify_data, sp)

    if song_center is None:
        return []

    scaler             = pipeline.steps[0][1]
    scaled_data        = scaler.transform(spotify_data[NUMBER_COLS])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))

    distances = cdist(scaled_song_center, scaled_data, "cosine")
    top_indices = list(np.argsort(distances)[:, :n_songs][0])

    recommendations = spotify_data.iloc[top_indices]
    recommendations = recommendations[~recommendations["name"].isin(song_dict["name"])]

    return recommendations[METADATA_COLS].to_dict(orient="records")


# ==============================================================================
# Fonctions internes
# ==============================================================================

def _get_song_data(song: dict, spotify_data: pd.DataFrame, sp: spotipy.Spotify) -> pd.Series | None:
    """
    Récupère les données d'une chanson depuis le dataset local,
    ou depuis l'API Spotify si elle est absente.

    Paramètres :
        song         : Dictionnaire {'name': str, 'year': int}
        spotify_data : DataFrame principal des chansons
        sp           : Client Spotify authentifié

    Retourne :
        Series des features audio, ou None si introuvable
    """
    try:
        return spotify_data[
            (spotify_data["name"] == song["name"]) &
            (spotify_data["year"] == song["year"])
        ].iloc[0]
    except IndexError:
        return _find_song_on_spotify(song["name"], song["year"], sp)


def _find_song_on_spotify(name: str, year: int, sp: spotipy.Spotify) -> pd.DataFrame | None:
    """
    Recherche une chanson sur l'API Spotify et retourne ses features audio.

    Paramètres :
        name : Nom de la chanson
        year : Année de la chanson
        sp   : Client Spotify authentifié

    Retourne :
        DataFrame d'une ligne avec les features audio, ou None si introuvable
    """
    try:
        results = sp.search(q=f"track:{name} year:{year}", limit=1)

        if not results["tracks"]["items"]:
            return None

        track          = results["tracks"]["items"][0]
        track_id       = track["id"]
        audio_features = sp.audio_features(track_id)

        if not audio_features or audio_features[0] is None:
            return None

        audio_features = audio_features[0]

        song_data = {
            "name":        [name],
            "year":        [year],
            "explicit":    [int(track["explicit"])],
            "duration_ms": [track["duration_ms"]],
            "popularity":  [track["popularity"]],
            **{key: value for key, value in audio_features.items()}
        }

        return pd.DataFrame(song_data)

    except spotipy.exceptions.SpotifyException as e:
        st.warning(f"Erreur API Spotify : {e}")
        return None
    except Exception as e:
        st.warning(f"Erreur inattendue lors de la recherche : {e}")
        return None


def _get_mean_vector(
    song_list: list,
    spotify_data: pd.DataFrame,
    sp: spotipy.Spotify
) -> np.ndarray | None:
    """
    Calcule le vecteur moyen des features audio pour une liste de chansons.

    Paramètres :
        song_list    : Liste de dictionnaires {'name': str, 'year': int}
        spotify_data : DataFrame principal des chansons
        sp           : Client Spotify authentifié

    Retourne :
        Vecteur numpy moyen, ou None si aucune chanson n'a pu être trouvée
    """
    song_vectors = []

    for song in song_list:
        song_data = _get_song_data(song, spotify_data, sp)

        if song_data is None:
            st.warning(f"« {song['name']} » est introuvable dans le dataset ou sur Spotify.")
            continue

        song_vectors.append(song_data[NUMBER_COLS].values)

    if not song_vectors:
        return None

    return np.mean(np.array(song_vectors), axis=0)


def _flatten_dict_list(dict_list: list) -> dict:
    """
    Regroupe une liste de dictionnaires en un dictionnaire de listes.

    Exemple :
        [{'name': 'A', 'year': 2020}, {'name': 'B', 'year': 2019}]
        → {'name': ['A', 'B'], 'year': [2020, 2019]}

    Paramètres :
        dict_list : Liste de dictionnaires avec les mêmes clés

    Retourne :
        Dictionnaire de listes
    """
    flattened = defaultdict(list)
    for item in dict_list:
        for key, value in item.items():
            flattened[key].append(value)
    return flattened