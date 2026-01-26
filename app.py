import streamlit as st
import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cdist
from collections import defaultdict
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# --- CONFIGURATION ---
st.set_page_config(page_title="Spotify Recommender & Analysis", layout="wide")

CLIENT_ID = st.secrets["spotify"]["client_id"]
CLIENT_SECRET = st.secrets["spotify"]["client_secret"]
                                      
try:
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
except Exception as e:
    st.error(f"Erreur de connexion à l'API Spotify : {e}")
    st.stop()


sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, 
                                                           client_secret=CLIENT_SECRET))

# --- CHARGEMENT DES DONNÉES (AVEC CACHE) ---
@st.cache_data
def load_data():
    # Chemins relatifs vers le dossier 'data'
    data = pd.read_csv("data/data.csv")
    genre_data = pd.read_csv("data/data_by_genres.csv")
    year_data = pd.read_csv("data/data_by_year.csv")
    artist_data = pd.read_csv("data/data_by_artist.csv")
    
    # Création de la colonne decade
    data['decade'] = data['year'].apply(lambda year : f'{(year//10)*10}s')
    
    return data, genre_data, year_data, artist_data

# --- ENTRAINEMENT DU MODELE (AVEC CACHE) ---
@st.cache_resource
def train_pipeline(spotify_data):
    # Préparation des données numériques
    X = spotify_data.select_dtypes(np.number)
    
    # Pipeline et KMeans
    song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                      ('kmeans', KMeans(n_clusters=25, verbose=False, random_state=42))], 
                                     verbose=False)
    song_cluster_pipeline.fit(X)
    
    return song_cluster_pipeline

# --- FONCTIONS DE RECOMMANDATION ---

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
               'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

def find_song(name, year):
    song_data = defaultdict()
    try:
        results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
        if results['tracks']['items'] == []:
            return None
        
        results = results['tracks']['items'][0]
        track_id = results['id']
        
        
        audio_features = sp.audio_features(track_id)
        if not audio_features or audio_features[0] is None:
            return None
            
        audio_features = audio_features[0]

        song_data['name'] = [name]
        song_data['year'] = [year]
        song_data['explicit'] = [int(results['explicit'])]
        song_data['duration_ms'] = [results['duration_ms']]
        song_data['popularity'] = [results['popularity']]

        for key, value in audio_features.items():
            song_data[key] = value
            
        return pd.DataFrame(song_data)
    
    except spotipy.exceptions.SpotifyException as e:
        st.warning(f"Erreur API Spotify (limite atteinte ou clé invalide) : {e}")
        return None
    except Exception as e:
        st.warning(f"Erreur inattendue lors de la recherche : {e}")
        return None
def get_song_data(song, spotify_data):
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    except IndexError:
        return find_song(song['name'], song['year'])

def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            st.warning(f'Warning: {song["name"]} does not exist in Spotify or in database')
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0) 

def flatten_dict_list(dict_list):
    flattened_dict = defaultdict()
    for key in dict_list[0].keys(): 
        flattened_dict[key] = []
    for dic in dict_list:
        for key,value in dic.items():
            flattened_dict[key].append(value)
    return flattened_dict

def recommend_songs(song_list, spotify_data, pipeline, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    
    song_center = get_mean_vector(song_list, spotify_data)
    
    scaler = pipeline.steps[0][1] 
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

# --- INTERFACE STREAMLIT ---

def main():
    data, genre_data, year_data, artist_data = load_data()
    pipeline = train_pipeline(data)

    st.sidebar.title("Menu Principal")
    page = st.sidebar.radio("Aller à", ["Recommandation de Musique", "Analyse Exploratoire (EDA)", "Clustering"])

    if page == "Recommandation de Musique":
        st.header("🎵 Système de Recommandation")
        st.write("Entrez une chanson que vous aimez, nous vous en proposerons 10 similaires.")

        col1, col2 = st.columns(2)
        with col1:
            song_name = st.text_input("Nom de la chanson", "Dior")
        with col2:
            song_year = st.number_input("Année", 2000, 2030, 2019)

        if st.button("Recommander"):
            with st.spinner('Recherche en cours...'):
                results = recommend_songs([{'name': song_name, 'year': song_year}], data, pipeline)
                
            if results:
                st.success(f"Voici des recommandations basées sur '{song_name}' ({song_year}) :")
                df_results = pd.DataFrame(results)
                st.dataframe(df_results)
            else:
                st.error("Aucune recommandation trouvée. Vérifiez le nom de la chanson.")

    elif page == "Analyse Exploratoire (EDA)":
        st.header("📊 Analyse des Données")
        
        # Distribution par décennie
        st.subheader("Nombre de chansons par décennie")
        fig_decade = px.histogram(data, x='decade', title='Distribution des décennies')
        st.plotly_chart(fig_decade, use_container_width=True)

        # Caractéristiques au fil du temps
        st.subheader("Évolution des caractéristiques sonores")
        sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'valence']
        fig_line = px.line(year_data, x='year', y=sound_features, title='Tendance audio (1921-2020)')
        st.plotly_chart(fig_line, use_container_width=True)

        # Volume sonore
        fig_loudness = px.line(year_data, x='year', y='loudness', title='Évolution du volume (Loudness)')
        st.plotly_chart(fig_loudness, use_container_width=True)

        # Top genres
        st.subheader("Top 10 Genres les plus populaires")
        top10_genres = genre_data.nlargest(10, 'popularity')
        fig_bar = px.bar(top10_genres, x='genres', y=['valence', 'energy', 'danceability', 'acousticness'], 
                         barmode='group', title='Caractéristiques du Top 10')
        st.plotly_chart(fig_bar, use_container_width=True)

        # WordClouds
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Nuage de mots : Genres")
            stopwords = set(STOPWORDS)
            comment_words = " ".join(genre_data['genres'])+" "
            wordcloud = WordCloud(width=400, height=400, background_color='white', stopwords=stopwords, min_font_size=10).generate(comment_words)
            plt.figure(figsize=(4,4))
            plt.imshow(wordcloud)
            plt.axis("off")
            st.pyplot(plt)
            plt.clf() # Clear figure

        with col2:
            st.subheader("Nuage de mots : Artistes")
            comment_words_artists = " ".join(artist_data['artists'])+" "
            wordcloud_artists = WordCloud(width=400, height=400, background_color='white', stopwords=stopwords, min_word_length=3, min_font_size=10).generate(comment_words_artists)
            plt.figure(figsize=(4,4))
            plt.imshow(wordcloud_artists)
            plt.axis("off")
            st.pyplot(plt)
            plt.clf()

    elif page == "Clustering":
        st.header("🔮 Visualisation des Clusters")
        
        # Song Clusters (PCA pour la rapidité d'affichage)
        st.subheader("Projection PCA des Chansons")
        # On recalcule l'embedding pour l'affichage ( PCA au lieu de t-SNE pour aller plus vite sur le web)
        X = data.select_dtypes(np.number)
        # Récupérer le scaler déjà fitté
        scaler = pipeline.steps[0][1]
        scaled_X = scaler.transform(X)
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        song_embedding = pca.fit_transform(scaled_X)
        
        projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
        projection['title'] = data['name']
        # Récupérer les labels du K-Means déjà fitté
        kmeans = pipeline.steps[1][1]
        projection['cluster'] = kmeans.labels_

        fig_pca = px.scatter(projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'], 
                             title='Clusters des chansons (Projection PCA)', opacity=0.7)
        st.plotly_chart(fig_pca, use_container_width=True)

if __name__ == '__main__':
    main()