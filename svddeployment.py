import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD
from fuzzywuzzy import process


class MultivariateAnalysis:
    def __init__(self, data_path, n_components=20):
        self.data = pd.read_csv(data_path)
        self.n_components = n_components
        self.model = None
        self.trainset = None
        self.movie_titles = {}
        self.movie_mapper = {}
        self.movie_inv_mapper = {}
        self._prepare_data()

    def _prepare_data(self):
        movie_ids = self.data['movieId'].unique()
        self.movie_titles = self.data.drop_duplicates('movieId')[['movieId', 'title']].set_index('movieId')[
            'title'].to_dict()
        self.movie_mapper = {movie_id: i for i, movie_id in enumerate(movie_ids)}
        self.movie_inv_mapper = {i: movie_id for i, movie_id in enumerate(movie_ids)}

        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(self.data[['userId', 'movieId', 'avg_rating']], reader)
        trainset = data.build_full_trainset()

        self.model = SVD(n_factors=self.n_components)
        self.model.fit(trainset)
        self.trainset = trainset

    def find_similar_movies(self, movie_id, top_k=5):
        if movie_id not in self.movie_mapper:
            raise ValueError("Movie ID not found in the dataset.")

        movie_index = self.movie_mapper[movie_id]
        movie_vector = np.array(self.model.qi[movie_index]).reshape(1, -1)
        all_movie_vectors = np.array(self.model.qi)
        similarities = cosine_similarity(movie_vector, all_movie_vectors)
        similar_indices = similarities.argsort()[0][-top_k - 1:-1][::-1]

        similar_movies = [self.movie_titles[self.movie_inv_mapper[idx]] for idx in similar_indices]
        return similar_movies

    def plot_similarity_heatmap(self, movie_id, top_k=5):
        if movie_id not in self.movie_mapper:
            raise ValueError("Movie ID not found in the dataset.")

        movie_index = self.movie_mapper[movie_id]
        movie_vector = np.array(self.model.qi[movie_index]).reshape(1, -1)
        all_movie_vectors = np.array(self.model.qi)
        similarities = cosine_similarity(movie_vector, all_movie_vectors)
        similar_indices = similarities.argsort()[0][-top_k - 1:-1][::-1]

        selected_indices = [movie_index] + similar_indices.tolist()
        selected_vectors = np.array([self.model.qi[idx] for idx in selected_indices])
        selected_titles = [self.movie_titles[self.movie_inv_mapper[idx]] for idx in selected_indices]

        similarity_matrix = cosine_similarity(selected_vectors)

        plt.figure(figsize=(12, 8))
        sns.heatmap(similarity_matrix, xticklabels=selected_titles, yticklabels=selected_titles, annot=True, fmt='.2f',
                    cmap='coolwarm')
        plt.title(f'Similarity Heatmap of "{self.movie_titles[movie_id]}" and Similar Movies')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        st.pyplot(plt)


if __name__ == "__main__":
    st.title("🍿 SVD-Based Movie Recommendation System 🎥")

    st.sidebar.header("Movie Recommendation Settings")
    st.sidebar.markdown("Use the input below to search for a movie title and get recommendations.")

    data_path = 'data/cleaned-data.csv'
    analysis = MultivariateAnalysis(data_path, n_components=20)

    movie_title_input = st.sidebar.text_input("Enter the title of a movie:", "Toy Story")

    if movie_title_input:
        closest_match = process.extractOne(movie_title_input, list(analysis.movie_titles.values()))
        closest_movie_title = closest_match[0]
        movie_id = list(analysis.movie_titles.keys())[list(analysis.movie_titles.values()).index(closest_movie_title)]

        similar_movies = analysis.find_similar_movies(movie_id)

        st.subheader(f"Movies similar to '{closest_movie_title}':")
        for movie in similar_movies:
            st.markdown(f"- {movie}")

        st.subheader("Similarity Heatmap:")
        analysis.plot_similarity_heatmap(movie_id)
