import streamlit as st
from fuzzywuzzy import process
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from main import DLPreprocessing, DLModeling, DLEvaluation, HDLPreprocessing, HDLModeling, HDLContentBasedFiltering, HDLEvaluation, HybridRecommendation


class HybridRecommendation:
    def __init__(self, cf_model, cb_model, data, movie_tfidf_matrix):
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.data = data
        self.movie_tfidf_matrix = movie_tfidf_matrix

    def recommend(self, user_id=None, movie_title=None, top_k=5):
        if user_id is not None:
            movie_ids = np.arange(self.cf_model.input[1].shape[1])
            user_ids = np.full(self.cf_model.input[0].shape[1], user_id)
            preds = self.cf_model.predict([user_ids, movie_ids])
            top_k_movie_indices = np.argsort(preds.flatten())[::-1][:top_k]
            return self.data['title'].iloc[top_k_movie_indices].unique(), preds.flatten()[top_k_movie_indices][:top_k]
        elif movie_title is not None:
            movie_title = process.extractOne(movie_title, self.data['title'])[0]
            movie_id = self.data[self.data['title'] == movie_title].index[0]
            movie_vector = self.movie_tfidf_matrix[movie_id]
            cosine_sim = cosine_similarity(movie_vector, self.movie_tfidf_matrix).flatten()
            sim_scores = list(enumerate(cosine_sim))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            unique_titles = set()
            top_k_movie_indices = []
            for i, score in sim_scores:
                if len(unique_titles) >= top_k:
                    break
                title = self.data['title'].iloc[i]
                if title not in unique_titles:
                    unique_titles.add(title)
                    top_k_movie_indices.append(i)

            return self.data['title'].iloc[top_k_movie_indices], [sim_scores[i] for i in top_k_movie_indices]
        else:
            raise ValueError("Either user_id or movie_title must be provided.")


# Filepath to the CSV file
filepath = 'data/cleaned-data.csv'

# Preprocessing
hdl_preprocessing = HDLPreprocessing(filepath)
data = hdl_preprocessing.load_data()
train_data, test_data, movie_tfidf_matrix = hdl_preprocessing.preprocess_data()

# Collaborative Filtering Modeling
hdl_modeling = HDLModeling(hdl_preprocessing.user_count, hdl_preprocessing.movie_count)
hdl_modeling.build_model()
hdl_modeling.train_model(train_data, epochs=10, batch_size=64)
cf_model = hdl_modeling.get_model()

# Content-Based Filtering for Cold Start
hdl_cb_filtering = HDLContentBasedFiltering(movie_tfidf_matrix, data)

# Hybrid Recommendation System
hybrid_recommender = HybridRecommendation(cf_model, hdl_cb_filtering, data, movie_tfidf_matrix)

# Streamlit app
st.title("Hybrid Recommendation System")
st.write("Get movie recommendations based on user ID or movie title")

option = st.selectbox("Choose Recommendation Type", ("User ID", "Movie Title"))

if option == "User ID":
    user_id = st.number_input("Enter User ID", min_value=0, step=1)
    if st.button("Recommend Movies"):
        recommendations, scores = hybrid_recommender.recommend(user_id=user_id, top_k=5)
        st.write("Top 5 recommendations for User ID", user_id)
        for i, (movie, score) in enumerate(zip(recommendations, scores), 1):
            st.write(f"{i}. {movie} - Predicted Rating: {score}")

elif option == "Movie Title":
    movie_title = st.text_input("Enter Movie Title")
    if st.button("Recommend Movies"):
        recommendations, scores = hybrid_recommender.recommend(movie_title=movie_title, top_k=5)
        st.write("Top 5 recommendations for Movie Title", movie_title)
        for i, (movie, score) in enumerate(zip(recommendations, scores), 1):
            st.write(f"{i}. {movie} - Similarity Score: {score[1]}")
