# ==============================================================================
# Module : Chargement des données
# ==============================================================================
# Objectif du module :
#     Ce module fournit les fonctions de chargement des fichiers CSV
#     constituant le dataset Spotify.
#
#     Il effectue les actions suivantes :
#     - Charge les quatre sources de données depuis le dossier 'data/'.
#     - Crée la colonne 'decade' pour regrouper les chansons par décennie.
#
# Exemple d'utilisation :
#     from src.data_loader import load_data
# ==============================================================================


import pandas as pd


# ==============================================================================
# Constantes
# ==============================================================================

DATA_PATH        = "data/data.csv"
GENRE_DATA_PATH  = "data/data_by_genres.csv"
YEAR_DATA_PATH   = "data/data_by_year.csv"
ARTIST_DATA_PATH = "data/data_by_artist.csv"


# ==============================================================================
# Chargement des données
# ==============================================================================

def load_data() -> tuple:
    """
    Charge les quatre fichiers CSV du dataset Spotify
    et enrichit le DataFrame principal avec la colonne 'decade'.

    Retourne :
        Tuple (data, genre_data, year_data, artist_data)
            - data        : DataFrame principal des chansons
            - genre_data  : DataFrame agrégé par genre
            - year_data   : DataFrame agrégé par année
            - artist_data : DataFrame agrégé par artiste
    """
    data        = pd.read_csv(DATA_PATH)
    genre_data  = pd.read_csv(GENRE_DATA_PATH)
    year_data   = pd.read_csv(YEAR_DATA_PATH)
    artist_data = pd.read_csv(ARTIST_DATA_PATH)

    data = _add_decade_column(data)

    return data, genre_data, year_data, artist_data


# ==============================================================================
# Fonctions internes
# ==============================================================================

def _add_decade_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute une colonne 'decade' indiquant la décennie de chaque chanson.

    Paramètres :
        df : DataFrame des chansons avec une colonne 'year'

    Retourne :
        DataFrame enrichi de la colonne 'decade' (ex. '1990s', '2000s')
    """
    df["decade"] = df["year"].apply(lambda year: f"{(year // 10) * 10}s")
    return df