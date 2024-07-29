# Import Libraries
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from surprise import Reader, Dataset, SVD
from wordcloud import WordCloud
from scipy import sparse
from surprise import Reader, Dataset
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from surprise.model_selection import cross_validate
from sklearn.decomposition import TruncatedSVD
from surprise.prediction_algorithms import SVD
from surprise.prediction_algorithms import KNNWithMeans, KNNBasic, KNNBaseline
from surprise.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import pickle


class MovieLensDataExplorer:
    def __init__(self, ratings_path, movies_path, links_path=None, tags_path=None):
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)
        self.links = pd.read_csv(links_path) if links_path else None
        self.tags = pd.read_csv(tags_path) if tags_path else None

    def show_info(self):
        print("## Ratings Data:")
        print(self.ratings.info())
        print("\n## Movies Data:")
        print(self.movies.info())
        if self.links is not None:
            print("\n## Links Data:")
            print(self.links.info())
        if self.tags is not None:
            print("\n## Tags Data:")
            print(self.tags.info())

    def overview(self):
        for df_name, df in zip(['ratings', 'movies', 'links', 'tags'],
                               [self.ratings, self.movies, self.links, self.tags]):
            if df is None:
                continue
            print(f"\n## {df_name.capitalize()} Data Overview:")
            print(df.head().to_markdown(index=False, numalign="left", stralign="left"))  # Show first few rows
            print(f"\nShape: {df.shape}")

            # Unique Values and Missing Values
            unique_counts = df.apply(lambda x: len(set(x.dropna())))
            missing_counts = df.isnull().sum()
            missing_percent = (missing_counts / len(df)) * 100

            summary_df = pd.DataFrame({
                'Unique Values': unique_counts,
                'Missing Values': missing_counts,
                '% Missing': missing_percent
            })
            print(summary_df.to_markdown(numalign="left", stralign="left"))

    def visualize(self):
        # Rating Distribution
        plt.figure(figsize=(8, 5))
        sns.countplot(x='rating', data=self.ratings, hue='rating', dodge=False, palette="viridis")
        plt.legend([], [], frameon=False)
        plt.title("Distribution of Movie Ratings")
        plt.show()

        # Number of Ratings per User and Movie
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(self.ratings.groupby('userId')['rating'].count(), bins=30, kde=True)
        plt.title("Number of Ratings per User")
        plt.subplot(1, 2, 2)
        sns.histplot(self.ratings.groupby('movieId')['rating'].count(), bins=30, kde=True)
        plt.title("Number of Ratings per Movie")
        plt.show()

        # Genre Distribution (if movies data is available)
        if self.movies is not None:
            all_genres = [genre for genres in self.movies['genres'].str.split('|') for genre in genres]
            genre_counts = pd.Series(all_genres).value_counts()

            plt.figure(figsize=(10, 6))

            # Create a custom color palette (optional, but enhances visual clarity)
            palette = sns.color_palette("husl", len(genre_counts))

            # Plot the barplot, assigning a specific color to each genre from the palette
            bars = plt.bar(genre_counts.index, genre_counts.values, color=palette)

            # Manually create legend handles and labels
            legend_elements = [plt.Rectangle((0, 0), 1, 1, color=palette[i], label=label)
                               for i, label in enumerate(genre_counts.index)]

            plt.xticks(rotation=70)
            plt.xlabel("Genres")
            plt.title("Genre Distribution")

            # Place the legend
            plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            plt.show()


class MovieLensDataCleaner:
    def __init__(self, ratings_path, movies_path, min_reviews):
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)
        self.min_reviews = min_reviews
        self.cleaned_df = None
        self.result_df = None

    def clean_and_merge(self):
        # Merge DataFrames
        df = pd.merge(self.ratings, self.movies, on='movieId')

        # Drop Timestamp
        df.drop('timestamp', axis=1, inplace=True)

        # Drop Duplicates
        df.drop_duplicates(inplace=True)

        # Genre Cleaning
        df['genres'] = df['genres'].str.split('|')

        # Extract Title and Year
        df['year'] = df['title'].str.extract(r'\((\d{4})\)', expand=False)
        df['year'] = pd.to_numeric(df['year'], errors='coerce')

        # Impute missing year values with median before dropping NaNs
        median_year = df['year'].median()
        df['year'] = df['year'].fillna(median_year).astype(int)

        df['title'] = df['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()

        # Drop NaN values for all columns (including year) after imputation
        self.cleaned_df = df.dropna()

        return self.cleaned_df

    def calculate_bayesian_average(self):
        if self.cleaned_df is None:
            raise ValueError("Data not cleaned. Call 'clean_and_merge' before calculating Bayesian average.")

        # Calculate mean rating and number of ratings for each movie
        movie_stats = self.cleaned_df.groupby('movieId').agg({'rating': ['mean', 'count']})
        movie_stats.columns = ['mean', 'count']

        # Calculate the global mean rating
        global_mean = self.cleaned_df['rating'].mean()

        # Calculate the Bayesian average
        def bayesian_avg(row):
            m = self.min_reviews
            v = row['count']
            r = row['mean']
            return (v / (v + m)) * r + (m / (v + m)) * global_mean

        movie_stats['bayesian_avg'] = movie_stats.apply(bayesian_avg, axis=1)

        # Join the Bayesian average with the cleaned dataframe
        self.result_df = self.cleaned_df.join(movie_stats['bayesian_avg'], on='movieId')
        self.result_df = self.result_df.rename(columns={"bayesian_avg": "avg_rating"})

        # Drop unnecessary columns
        self.result_df = self.result_df.drop(columns=['mean', 'count'], errors='ignore')

        return self.result_df

    def save_cleaned_data(self):
        if self.result_df is None:
            raise ValueError("Data not processed. Call 'calculate_bayesian_average' before saving cleaned data.")

        # Drop the 'rating' column
        result_df = self.result_df.drop(columns=['rating'], errors='ignore')

        # Save the cleaned data to CSV in the data folder, overwriting if exists
        output_path = os.path.join('data', 'cleaned-data.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create directory if it doesn't exist
        result_df.to_csv(output_path, index=False, mode='w')  # Use mode='w' to overwrite
        print(f"Cleaned data saved to: {output_path}")


class MovieLensUnivariateEDA:
    def __init__(self, data_path='data/cleaned-data.csv'):
        self.df = pd.read_csv(data_path)

    def analyze_ratings(self):
        print("### Rating Distribution Analysis:")

        # Histogram of avg_rating
        plt.figure(figsize=(8, 6))
        sns.histplot(self.df['avg_rating'], bins=10, kde=True, color='skyblue')
        plt.title('Distribution of Average Movie Ratings')
        plt.xlabel('Average Rating')
        plt.ylabel('Frequency')
        plt.show()

    def analyze_genres(self):
        print("\n### Genre Analysis:")

        # Flatten and Count Genres
        all_genres = [genre for genres in self.df['genres'].apply(eval) for genre in genres]
        genre_counts = pd.Series(all_genres).value_counts()

        # Bar plot of Top 10 and Least 10 Genres
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        sns.barplot(x=genre_counts[:10].index, y=genre_counts[:10].values, ax=axes[0], hue=genre_counts[:10].index,
                    dodge=False, palette='viridis')
        axes[0].legend([], [], frameon=False)  # Remove legend
        axes[0].set_title('Top 10 Movie Genres by Frequency')
        axes[0].set_xlabel('Genre')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        sns.barplot(x=genre_counts[-10:].index, y=genre_counts[-10:].values, ax=axes[1], hue=genre_counts[-10:].index,
                    dodge=False, palette='viridis')
        axes[1].legend([], [], frameon=False)  # Remove legend
        axes[1].set_title('Least 10 Movie Genres by Frequency')
        axes[1].set_xlabel('Genre')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)
        plt.show()

        # Word cloud of Genres (Larger and more readable)
        wordcloud = WordCloud(width=1200, height=600, background_color='white',
                              min_font_size=10).generate_from_frequencies(genre_counts)
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title('Word Cloud of Movie Genres')
        plt.show()

    def analyze_popularity(self):
        print("\n### Movie Popularity Analysis:")

        # Calculate rating counts per movie
        movie_rating_counts = self.df['movieId'].value_counts()

        # Simple scatterplot (without genre hue)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=movie_rating_counts, y=self.df.groupby('movieId')['avg_rating'].mean(), color='skyblue',
                        alpha=0.5)
        plt.title('Average Rating vs. Number of Ratings per Movie')
        plt.xlabel('Number of Ratings')
        plt.ylabel('Average Rating')
        plt.show()

    def analyze_years(self):
        print("\n### Year Analysis:")

        # Calculate 4-year ranges
        min_year = int(self.df['year'].min())
        max_year = int(self.df['year'].max())
        bins = range(min_year, max_year + 5, 4)  # Create bins in 4-year intervals

        # Create year_range column
        self.df['year_range'] = pd.cut(self.df['year'], bins=bins, labels=False) * 4 + min_year

        # Bar chart of Number of Movies per Year Range
        year_range_counts = self.df['year_range'].value_counts().sort_index()
        plt.figure(figsize=(12, 6))
        sns.barplot(x=year_range_counts.index, y=year_range_counts.values, color='skyblue')
        plt.title('Number of Movies per 4-Year Range')
        plt.xlabel('Year Range')
        plt.ylabel('Number of Movies')
        plt.xticks(rotation=45)
        plt.show()

    def analyze_top_bottom(self, top_n=5):
        print("\n### Top and Bottom Rated Movies & Users:")

        # Top and Bottom Movies by Average Rating
        top_movies = self.df.groupby('title')['avg_rating'].mean().nlargest(top_n)
        bottom_movies = self.df.groupby('title')['avg_rating'].mean().nsmallest(top_n)

        print("\nTop Rated Movies:")
        print(top_movies.to_markdown(numalign='left', stralign='left'))
        print("\nBottom Rated Movies:")
        print(bottom_movies.to_markdown(numalign='left', stralign='left'))

        # Top and Bottom Users by Average Rating
        top_users = self.df.groupby('userId')['avg_rating'].mean().nlargest(top_n)
        bottom_users = self.df.groupby('userId')['avg_rating'].mean().nsmallest(top_n)

        print("\nTop Raters (By Average Rating):")
        print(top_users.to_markdown(numalign='left', stralign='left'))
        print("\nBottom Raters (By Average Rating):")
        print(bottom_users.to_markdown(numalign='left', stralign='left'))

        # Top and Bottom Users by Number of Ratings (Bar Plots)
        user_rating_counts = self.df.groupby('userId')['avg_rating'].count()
        top_raters_by_count = user_rating_counts.nlargest(top_n)
        bottom_raters_by_count = user_rating_counts.nsmallest(top_n)

        print("\nTop Raters (By Number of Ratings):")
        print(top_raters_by_count.to_markdown(numalign='left', stralign='left'))
        print("\nBottom Raters (By Number of Ratings):")
        print(bottom_raters_by_count.to_markdown(numalign='left', stralign='left'))


class MovieLensBivariateEDA:
    def __init__(self, data_path='data/cleaned-data.csv'):
        self.df = pd.read_csv(data_path)
        self.df['genres'] = self.df['genres'].apply(eval)  # Ensure genres are lists
        self._prepare_genre_columns()

    def _prepare_genre_columns(self):
        # Get a set of all unique genres
        unique_genres = set(g for genres in self.df['genres'] for g in genres)

        # Create a column for each genre
        for genre in unique_genres:
            self.df[genre] = self.df['genres'].apply(lambda x: 1 if genre in x else 0)

    def analyze_year_vs_avg_rating(self):
        print("\n### Year vs. Average Rating Analysis:")

        # Calculate average rating for each year
        year_avg_ratings = self.df.groupby('year')['avg_rating'].mean()

        # Plotting
        plt.figure(figsize=(12, 8))
        sns.lineplot(x=year_avg_ratings.index, y=year_avg_ratings.values, marker='o', color='skyblue')
        plt.title('Average Rating by Year')
        plt.xlabel('Year')
        plt.ylabel('Average Rating')
        plt.show()

    def analyze_genres_vs_avg_rating(self):
        print("\n### Genres vs. Average Rating Analysis:")

        # Create a long-form DataFrame for Seaborn
        genre_ratings = []
        for genre in self.df.columns:
            if genre not in ['userId', 'movieId', 'title', 'genres', 'year', 'avg_rating']:
                genre_data = self.df[self.df[genre] == 1]
                for rating in genre_data['avg_rating']:
                    genre_ratings.append((genre, rating))

        genre_ratings_df = pd.DataFrame(genre_ratings, columns=['Genre', 'Average Rating'])

        # Plotting
        plt.figure(figsize=(15, 8))
        sns.stripplot(x='Genre', y='Average Rating', data=genre_ratings_df, jitter=True, hue='Genre', palette='viridis',
                      alpha=0.6, legend=False)
        plt.title('Average Rating Distribution by Genre')
        plt.xlabel('Genre')
        plt.ylabel('Average Rating')
        plt.xticks(rotation=45)
        plt.show()


class MultivariateAnalysis:
    def __init__(self, data_path, n_components=20):
        """
        Initializes the MultivariateAnalysis class.

        Parameters:
        - data_path: Path to the CSV file containing the data.
        - n_components: Number of latent features for dimensionality reduction.
        """
        self.data = pd.read_csv(data_path)
        self.n_components = n_components
        self.model = None
        self.trainset = None
        self.movie_titles = {}
        self.movie_mapper = {}
        self.movie_inv_mapper = {}
        self._prepare_data()

    def _prepare_data(self):
        """
        Prepares the data for SVD model.
        """
        # Create a mapping of movieId to title and index
        movie_ids = self.data['movieId'].unique()
        self.movie_titles = self.data.drop_duplicates('movieId')[['movieId', 'title']].set_index('movieId')[
            'title'].to_dict()
        self.movie_mapper = {movie_id: i for i, movie_id in enumerate(movie_ids)}
        self.movie_inv_mapper = {i: movie_id for i, movie_id in enumerate(movie_ids)}

        # Prepare data for surprise
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(self.data[['userId', 'movieId', 'avg_rating']], reader)
        trainset = data.build_full_trainset()

        # Initialize and train the SVD model
        self.model = SVD(n_factors=self.n_components)
        self.model.fit(trainset)
        self.trainset = trainset

    def find_similar_movies(self, movie_id, top_k=5):
        """
        Finds similar movies to a given movie.

        Parameters:
        - movie_id: The ID of the movie to find similar movies for.
        - top_k: The number of similar movies to return.

        Returns:
        - A list of movie titles that are similar to the given movie.
        """
        if movie_id not in self.movie_mapper:
            raise ValueError("Movie ID not found in the dataset.")

        movie_index = self.movie_mapper[movie_id]
        movie_vector = np.array(self.model.qi[movie_index]).reshape(1, -1)
        all_movie_vectors = np.array(self.model.qi)
        similarities = cosine_similarity(movie_vector, all_movie_vectors)
        similar_indices = similarities.argsort()[0][-top_k - 1:-1][::-1]  # Exclude the movie itself and sort

        similar_movies = [self.movie_titles[self.movie_inv_mapper[idx]] for idx in similar_indices]
        return similar_movies

    def plot_similarity_heatmap(self, movie_id, top_k=5):
        """
        Plots a heatmap of similarities between a movie and its similar movies.

        Parameters:
        - movie_id: The ID of the movie to find similar movies for.
        - top_k: The number of similar movies to return in the heatmap.
        """
        if movie_id not in self.movie_mapper:
            raise ValueError("Movie ID not found in the dataset.")

        movie_index = self.movie_mapper[movie_id]
        movie_vector = np.array(self.model.qi[movie_index]).reshape(1, -1)
        all_movie_vectors = np.array(self.model.qi)
        similarities = cosine_similarity(movie_vector, all_movie_vectors)
        similar_indices = similarities.argsort()[0][-top_k - 1:-1][::-1]  # Exclude the movie itself and sort

        selected_indices = [movie_index] + similar_indices.tolist()
        selected_vectors = np.array([self.model.qi[idx] for idx in selected_indices])
        selected_titles = [self.movie_titles[self.movie_inv_mapper[idx]] for idx in selected_indices]

        similarity_matrix = cosine_similarity(selected_vectors)

        plt.figure(figsize=(12, 8))
        sns.heatmap(similarity_matrix, xticklabels=selected_titles, yticklabels=selected_titles, annot=True, fmt='.2f',
                    cmap='coolwarm')
        plt.title(f'Similarity Heatmap of "{self.movie_titles[movie_id]}" and Similar Movies')
        plt.show()


class DataPreprocessing:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = pd.read_csv(filepath)

    def preprocess(self):
        # Convert genres from string representation to list
        self.data['genres'] = self.data['genres'].apply(eval)

        # Create user-item matrix
        self.X, self.user_mapper, self.movie_mapper, self.user_inv_mapper, self.movie_inv_mapper = self.create_user_item_matrix()

        # Create genre matrix
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


class CollaborativeFiltering:
    def __init__(self, X, movie_titles, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper):
        self.X = X
        self.movie_titles = movie_titles
        self.user_mapper = user_mapper
        self.movie_mapper = movie_mapper
        self.user_inv_mapper = user_inv_mapper
        self.movie_inv_mapper = movie_inv_mapper

    def find_similar_movies(self, movie_id, k=5, metric='cosine'):
        """
        Finds similar movies to a given movie using collaborative filtering.

        Parameters:
        - movie_id: The ID of the movie to find similar movies for.
        - k: The number of similar movies to return.
        - metric: The distance metric to use ('cosine' or 'euclidean').

        Returns:
        - A list of movie titles that are similar to the given movie.
        """
        X = self.X.T

        if movie_id not in self.movie_mapper:
            raise ValueError("Movie ID not found in the dataset.")

        movie_ind = self.movie_mapper[movie_id]
        movie_vec = X[movie_ind].reshape(1, -1)

        # Initialize NearestNeighbors
        kNN = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric=metric)
        kNN.fit(X)

        # Find neighbors
        distances, indices = kNN.kneighbors(movie_vec, return_distance=True)

        # Exclude the movie itself
        similar_indices = indices[0][1:]

        # Retrieve movie IDs and titles
        similar_movie_ids = [self.movie_inv_mapper[idx] for idx in similar_indices]
        similar_movie_titles = [self.movie_titles[movie_id] for movie_id in similar_movie_ids]

        return similar_movie_titles


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
                print("Using content-based filtering due to insufficient user ratings.")
                return self.get_content_based_recommendations(title_string, n_recommendations)
            else:
                print("Using collaborative filtering based on user history.")
                return self.get_collaborative_recommendations(movie_id, n_recommendations)
        else:
            rated_movie_indices = np.where(self.X[:, self.movie_mapper[movie_id]].toarray().flatten() > 0)[0]

            if len(rated_movie_indices) < self.movie_threshold:
                print("Using content-based filtering due to insufficient movie ratings.")
                return self.get_content_based_recommendations(title_string, n_recommendations)
            else:
                print("Using collaborative filtering based on movie similarities.")
                return self.get_collaborative_recommendations(movie_id, n_recommendations)



