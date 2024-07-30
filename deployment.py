import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process


# Data Preprocessing Class
class DataPreprocessing:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = pd.read_csv(filepath)

    def preprocess(self):
        self.data['genres'] = self.data['genres'].apply(eval)
        self.X, self.user_mapper, self.movie_mapper, self.user_inv_mapper, self.movie_inv_mapper = self.create_user_item_matrix()
        self.genre_matrix = self.create_genre_matrix()
        return self.X, self.genre_matrix, self.user_mapper, self.movie_mapper, self.user_inv_mapper, self.movie_inv_mapper

    def create_user_item_matrix(self):
        M = self.data['userId'].nunique()
        N = self.data['movieId'].nunique()

        user_mapper = dict(zip(np.unique(self.data["userId"]), list(range(M))))
        movie_mapper = dict(zip(np.unique(self.data["movieId"]), list(range(N))))

        user_inv_mapper = dict(zip(list(range(M)), np.unique(self.data["userId"])))
        movie_inv_mapper = dict(zip(list(range(N)), np.unique(self.data["movieId"])))

        user_index = [user_mapper[i] for i in self.data['userId']]
        item_index = [movie_mapper[i] for i in self.data['movieId']]

        X = csr_matrix((self.data["avg_rating"], (user_index, item_index)), shape=(M, N))

        return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

    def create_genre_matrix(self):
        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(self.data['genres'])
        return genre_matrix


# Recommender Class
class HybridRecommender:
    def __init__(self, X, genre_matrix, user_mapper, movie_mapper, movie_titles, user_inv_mapper, movie_inv_mapper,
                 user_threshold=5, movie_threshold=5):
        self.X = X
        self.genre_matrix = genre_matrix
        self.user_mapper = user_mapper
        self.movie_mapper = movie_mapper
        self.movie_titles = movie_titles
        self.user_inv_mapper = user_inv_mapper
        self.movie_inv_mapper = movie_inv_mapper
        self.user_threshold = user_threshold
        self.movie_threshold = movie_threshold

    def movie_finder(self, title):
        all_titles = self.movie_titles.tolist()
        closest_match = process.extractOne(title, all_titles)
        return closest_match[0]

    def get_content_based_recommendations(self, title_string, n_recommendations=10):
        title = self.movie_finder(title_string)
        idx = self.movie_titles[self.movie_titles == title].index[0]
        sim_scores = cosine_similarity(self.genre_matrix[idx].reshape(1, -1), self.genre_matrix).flatten()
        sim_scores = list(enumerate(sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        recommendations = []
        seen_titles = set()

        for i, score in sim_scores:
            movie_title = self.movie_titles.iloc[i]
            if movie_title not in seen_titles and movie_title != title:
                recommendations.append(movie_title)
                seen_titles.add(movie_title)
                if len(recommendations) >= n_recommendations:
                    break

        return recommendations

    def get_collaborative_recommendations(self, movie_id, k=5, metric='cosine'):
        X = self.X.T

        if movie_id not in self.movie_mapper:
            raise ValueError("Movie ID not found in the dataset.")

        movie_ind = self.movie_mapper[movie_id]
        movie_vec = X[movie_ind].reshape(1, -1)

        kNN = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric=metric)
        kNN.fit(X)

        distances, indices = kNN.kneighbors(movie_vec, return_distance=True)

        similar_indices = indices[0][1:]

        similar_movie_ids = [self.movie_inv_mapper[idx] for idx in similar_indices]
        similar_movie_titles = [self.movie_titles[self.movie_titles.index == movie_id].iloc[0] for movie_id in
                                similar_movie_ids]

        return similar_movie_titles

    def hybrid_recommendations(self, title_string, user_id=None, n_recommendations=10):
        title = self.movie_finder(title_string)
        movie_id = self.movie_titles[self.movie_titles == title].index[0]

        if user_id is not None and user_id in self.user_mapper:
            user_idx = self.user_mapper[user_id]
            user_ratings = self.X[user_idx].toarray().flatten()
            rated_movies = np.where(user_ratings > 0)[0]

            if len(rated_movies) < self.user_threshold:
                return self.get_content_based_recommendations(title_string, n_recommendations)
            else:
                return self.get_collaborative_recommendations(movie_id, n_recommendations)
        else:
            rated_movie_indices = np.where(self.X[:, self.movie_mapper[movie_id]].toarray().flatten() > 0)[0]

            if len(rated_movie_indices) < self.movie_threshold:
                return self.get_content_based_recommendations(title_string, n_recommendations)
            else:
                return self.get_collaborative_recommendations(movie_id, n_recommendations)


# Load and preprocess data
data_preprocessing = DataPreprocessing('data/cleaned-data.csv')
X, genre_matrix, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = data_preprocessing.preprocess()
movie_titles = data_preprocessing.data.set_index('movieId')['title']

# Streamlit App
st.title("Hybrid Movie Recommender System")
st.write("This application provides movie recommendations based on content and collaborative filtering.")

# User Inputs
title_input = st.text_input("Enter Movie Title", "Toy Story")
user_id_input = st.number_input("Enter User ID (optional)", value=None, format="%d")

# Hybrid recommender model
hybrid_model = HybridRecommender(X, genre_matrix, user_mapper, movie_mapper, movie_titles, user_inv_mapper,
                                 movie_inv_mapper)

if st.button("Get Recommendations"):
    recommendations = hybrid_model.hybrid_recommendations(title_string=title_input, user_id=user_id_input,
                                                          n_recommendations=5)

    st.subheader("Recommendations:")
    for rec in recommendations:
        st.write(rec)
