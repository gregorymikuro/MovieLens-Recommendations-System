# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

app = FastAPI()


# Define the HybridRecommender class
class HybridRecommender:
    def __init__(self, X, genre_matrix, user_mapper, movie_mapper, movie_titles):
        self.X = X
        self.genre_matrix = genre_matrix
        self.user_mapper = user_mapper
        self.movie_mapper = movie_mapper
        self.movie_titles = movie_titles

    def movie_finder(self, title):
        all_titles = self.movie_titles['title'].tolist()
        closest_match = process.extractOne(title, all_titles)
        return closest_match[0]

    def get_content_based_recommendations(self, title_string, n_recommendations=10):
        title = self.movie_finder(title_string)
        idx = self.movie_titles[self.movie_titles['title'] == title].index[0]
        sim_scores = cosine_similarity(self.genre_matrix[idx].reshape(1, -1), self.genre_matrix).flatten()
        sim_scores = list(enumerate(sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        recommendations = []
        seen_titles = set()

        for i, score in sim_scores:
            movie_title = self.movie_titles.iloc[i]['title']
            if movie_title not in seen_titles and movie_title != title:
                recommendations.append(movie_title)
                seen_titles.add(movie_title)
                if len(recommendations) >= n_recommendations:
                    break

        return recommendations

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            return HybridRecommender(
                X=data['X'],
                genre_matrix=data['genre_matrix'],
                user_mapper=data['user_mapper'],
                movie_mapper=data['movie_mapper'],
                movie_titles=data['movie_titles']
            )


# Load the model
model_file = 'models/hybrid_recommender.pkl'
if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model file {model_file} not found.")

model = HybridRecommender.load_model(model_file)


# Define the request model
class RecommendationRequest(BaseModel):
    title: str
    n_recommendations: int = 10


# Define API routes
@app.post("/recommendations/")
def get_recommendations(request: RecommendationRequest):
    try:
        recommendations = model.get_content_based_recommendations(request.title, request.n_recommendations)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the FastAPI app, use the following command:
# uvicorn app:app --reload
