# ==============================================================================
# Module : Système de recommandation musicale (Interface Streamlit)
# ==============================================================================
# Objectif du module :
#     Ce module fournit une interface interactive multi-pages permettant :
#     - De recommander des chansons similaires via un pipeline KMeans.
#     - D'explorer les données musicales (EDA) via des visualisations interactives.
#     - De visualiser les clusters de chansons par projection PCA.
#
#     Il effectue les actions suivantes :
#     - Charge les données CSV et entraîne le pipeline au démarrage (cache).
#     - Connecte l'API Spotify via les secrets Streamlit.
#     - Orchestre les trois pages de l'interface via une navigation latérale.
#
# Paramètres configurables (via l'interface) :
#     song_name : Nom de la chanson saisie par l'utilisateur
#     song_year : Année de la chanson saisie par l'utilisateur
#
# Prérequis :
#     Fichier .streamlit/secrets.toml avec les clés :
#         [spotify]
#         client_id     = "..."
#         client_secret = "..."
#
# Exemple d'utilisation :
#     streamlit run app.py
# ==============================================================================


import streamlit as st
import pandas as pd
import numpy as np
import spotipy
import plotly.express as px
import matplotlib.pyplot as plt

from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from collections import defaultdict
from wordcloud import WordCloud, STOPWORDS

from src.data_loader import load_data
from src.recommender import recommend_songs
from src.model import train_song_pipeline


# ==============================================================================
# Configuration de la page
# ==============================================================================

st.set_page_config(
    page_title="Spotify Recommender & Analysis",
    page_icon="🎵",
    layout="wide"
)


# ==============================================================================
# Constantes
# ==============================================================================

SOUND_FEATURES = [
    'acousticness', 'danceability', 'energy',
    'instrumentalness', 'valence'
]

PAGES = [
    "Recommandation de musique",
    "Analyse exploratoire (EDA)",
    "Clustering"
]


# ==============================================================================
# Connexion à l'API Spotify
# ==============================================================================

def connect_spotify() -> spotipy.Spotify:
    """
    Initialise et retourne un client Spotify authentifié
    via les secrets Streamlit.

    Retourne :
        Client Spotify authentifié

    Lève :
        st.stop() si la connexion échoue
    """
    try:
        auth_manager = SpotifyClientCredentials(
            client_id=st.secrets["spotify"]["client_id"],
            client_secret=st.secrets["spotify"]["client_secret"]
        )
        return spotipy.Spotify(auth_manager=auth_manager)
    except Exception as e:
        st.error(f"Erreur de connexion à l'API Spotify : {e}")
        st.stop()


# ==============================================================================
# Chargement des ressources (avec cache)
# ==============================================================================

@st.cache_resource
def load_resources() -> tuple:
    """
    Charge les données et entraîne le pipeline KMeans au démarrage.
    Résultat mis en cache pour éviter les rechargements.

    Retourne :
        Tuple (data, genre_data, year_data, artist_data, pipeline, client Spotify)
    """
    data, genre_data, year_data, artist_data = load_data()
    pipeline = train_song_pipeline(data)
    sp = connect_spotify()
    return data, genre_data, year_data, artist_data, pipeline, sp


# ==============================================================================
# Pages de l'interface
# ==============================================================================

def render_recommendation_page(data: pd.DataFrame, pipeline: Pipeline, sp: spotipy.Spotify) -> None:
    """
    Affiche la page de recommandation musicale.

    Paramètres :
        data     : DataFrame principal des chansons
        pipeline : Pipeline KMeans entraîné
        sp       : Client Spotify authentifié
    """
    st.header("🎵 Système de recommandation")
    st.write("Entrez une chanson que vous aimez, nous vous en proposerons 10 similaires.")

    col1, col2 = st.columns(2)
    with col1:
        song_name = st.text_input("Nom de la chanson", "Dior")
    with col2:
        song_year = st.number_input("Année", 2000, 2030, 2019)

    if st.button("Recommander"):
        with st.spinner("Recherche en cours..."):
            results = recommend_songs(
                song_list=[{"name": song_name, "year": song_year}],
                spotify_data=data,
                pipeline=pipeline,
                sp=sp
            )

        if results:
            st.success(f"Recommandations basées sur « {song_name} » ({song_year}) :")
            st.dataframe(pd.DataFrame(results))
        else:
            st.error("Aucune recommandation trouvée. Vérifiez le nom et l'année de la chanson.")


def render_eda_page(data: pd.DataFrame, genre_data: pd.DataFrame,
                    year_data: pd.DataFrame, artist_data: pd.DataFrame) -> None:
    """
    Affiche la page d'analyse exploratoire des données.

    Paramètres :
        data        : DataFrame principal des chansons
        genre_data  : DataFrame agrégé par genre
        year_data   : DataFrame agrégé par année
        artist_data : DataFrame agrégé par artiste
    """
    st.header("📊 Analyse des données")

    # -- Distribution par décennie --
    st.subheader("Nombre de chansons par décennie")
    fig_decade = px.histogram(data, x="decade", title="Distribution des décennies")
    st.plotly_chart(fig_decade, use_container_width=True)

    # -- Évolution des caractéristiques sonores --
    st.subheader("Évolution des caractéristiques sonores")
    fig_line = px.line(
        year_data, x="year", y=SOUND_FEATURES,
        title="Tendance audio (1921–2020)"
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # -- Volume sonore --
    fig_loudness = px.line(
        year_data, x="year", y="loudness",
        title="Évolution du volume sonore (Loudness)"
    )
    st.plotly_chart(fig_loudness, use_container_width=True)

    # -- Top 10 genres --
    st.subheader("Top 10 genres les plus populaires")
    top10_genres = genre_data.nlargest(10, "popularity")
    fig_bar = px.bar(
        top10_genres,
        x="genres",
        y=["valence", "energy", "danceability", "acousticness"],
        barmode="group",
        title="Caractéristiques audio du Top 10 des genres"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # -- Nuages de mots --
    col1, col2 = st.columns(2)
    stopwords = set(STOPWORDS)

    with col1:
        st.subheader("Nuage de mots : Genres")
        _render_wordcloud(" ".join(genre_data["genres"]), stopwords)

    with col2:
        st.subheader("Nuage de mots : Artistes")
        _render_wordcloud(" ".join(artist_data["artists"]), stopwords, min_word_length=3)


def render_clustering_page(data: pd.DataFrame, pipeline: Pipeline) -> None:
    """
    Affiche la page de visualisation des clusters par projection PCA.

    Paramètres :
        data     : DataFrame principal des chansons
        pipeline : Pipeline KMeans entraîné (scaler + kmeans)
    """
    st.header("🔮 Visualisation des clusters")
    st.subheader("Projection PCA des chansons")

    X            = data.select_dtypes(np.number)
    scaler       = pipeline.steps[0][1]
    scaled_X     = scaler.transform(X)

    pca            = PCA(n_components=2)
    song_embedding = pca.fit_transform(scaled_X)

    projection            = pd.DataFrame(song_embedding, columns=["x", "y"])
    projection["title"]   = data["name"].values
    projection["cluster"] = pipeline.steps[1][1].labels_

    fig = px.scatter(
        projection, x="x", y="y",
        color="cluster",
        hover_data=["title"],
        title="Clusters des chansons (Projection PCA)",
        opacity=0.7
    )
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# Fonctions utilitaires
# ==============================================================================

def _render_wordcloud(text: str, stopwords: set, min_word_length: int = 1) -> None:
    """
    Génère et affiche un nuage de mots à partir d'un texte.

    Paramètres :
        text             : Texte source (mots séparés par des espaces)
        stopwords        : Mots à exclure du nuage
        min_word_length  : Longueur minimale des mots à inclure
    """
    wordcloud = WordCloud(
        width=400, height=400,
        background_color="white",
        stopwords=stopwords,
        min_word_length=min_word_length,
        min_font_size=10
    ).generate(text)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(wordcloud)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


# ==============================================================================
# Point d'entrée
# ==============================================================================

if __name__ == "__main__":
    data, genre_data, year_data, artist_data, pipeline, sp = load_resources()

    st.sidebar.title("Menu principal")
    page = st.sidebar.radio("Aller à", PAGES)

    if page == PAGES[0]:
        render_recommendation_page(data, pipeline, sp)
    elif page == PAGES[1]:
        render_eda_page(data, genre_data, year_data, artist_data)
    elif page == PAGES[2]:
        render_clustering_page(data, pipeline)