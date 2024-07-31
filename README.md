# __Building a Movie Recommender with the MovieLens Dataset__

![alt text](image.png)

### 1.1 Business Overview

In the age of streaming services and an abundance of movie alternatives, customers frequently face the issue of finding films that match their preferences. This project aims to build a movie recommendation system that enhances the user’s movie-watching experience by suggesting films tailored to their preferences. The system will utilize machine learning algorithms and user data to deliver tailored movie suggestions based on ratings, viewing history, and preferences, resulting in growth and competitive advantage in the entertainment industry.
The target audience for this project is companies that provide movie streaming services, such as Netflix, Amazon Prime Video, or Hulu. These companies can, in turn, use recommendation systems to increase their customer engagement and retention.


### 1.2 Problem statement
In the vast and ever-expanding landscape of film and television content, users often face the challenge of discovering movies that align with their personal tastes and preferences. With countless options available across multiple platforms, finding enjoyable and relevant movies can be overwhelming and time-consuming. Traditional methods of browsing and searching are often inefficient, leading to decision fatigue and a suboptimal viewing experience. We aim to develop a Movie Recommender System that addresses this problem by providing users with personalized, relevant, and timely movie recommendations.

### 1.3 General Objective
To develop and implement a Movie Recommender System that delivers personalized and relevant movie suggestions to users, thereby improving their movie discovery experience and increasing their satisfaction with their entertainment choices.

### 1.4 Specific Objectives

1. Analyze and determine movie ratings considering the number of raters and the overall distribution of ratings by calculating the Bayesian average to ensure that ratings are representative and not overly influenced by the number of raters.

2. Investigate relationships between user preferences and movie features through matrix factorization techniques, such as Singular Value Decomposition (SVD) to help in understanding how latent factors can capture the underlying patterns in user-movie interactions.

3.  Create and deploy a hybrid recommendation system that integrates collaborative filtering with content-based filtering that addresses the cold start problem and optimizes recommendation accuracy by combining the strengths of both methods.
Data Understanding

### 1.5 Challenges
1. Cold start problem: Accurate recommendations will be complex to provide when there is limited data for new users or at the system's initial launch.
2. Scalability: Managing and processing large volumes of user data and movie metadata will be challenging as the system grows.
3. Algorithm Complexity: Balancing the complexity of recommendation algorithms with the need for real-time performance will be a challenge. More sophisticated algorithms might improve recommendation quality but could increase computational demands and processing time.
4. Diverse User Preferences: There will be issues when effectively handling a wide range of user tastes and preferences because failure to accommodate diverse preferences can result in recommendations that do not resonate with a large segment of users.
5. Handling Challenging Trends: Adapting to changes in user preferences and trends, such as seasonal interests or emerging genres, whereby recommendations may become outdated or irrelevant if the system cannot adapt quickly to changing user tastes.

Data preparation

The dataset ml-latest-small is from (https://grouplens.org/datasets/movielens/latest/)
It describes 5-star ratings and free-text tagging activity from movieLens, a movie recommendation system. The data contained 100836 ratings across 9742 movies and was generated on September 26th, 2018. Users were selected randomly for inclusion and all the users had rated at least 20 movies.

#### Summary of Features in the Dataset

* User_Id: A unique identifier for each user
* MovieId: A unique identifier for each movie and is consistent in the dataset in ratings, tags, movies, and links.
* TimeStamp: Represents seconds since midnight in Coordinated Universal Time (UTC).
* Tags: User-generated metadata about movies. Each tag is typically a single word or short phrase where each user determines a particular tag's meaning, value and purpose.
* Genre: Pipe-separated list, selected from Actions, Adventure, Animation, Children’s, Comedy, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fic, Thriller, War, Western, no genres listed.

#### _Citation_

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872


### Data Cleaning
Our data cleaning strategy aims to refine the dataset for subsequent analysis and modeling:

1.  **Merge `ratings` and `movies`:** We'll merge the `ratings` and `movies` DataFrames on the `movieId` column. This consolidation is essential for associating user ratings with specific movie titles and genres, enabling us to examine user preferences in relation to movie attributes and facilitating various recommendation strategies.

2.  **Timestamp Removal:** The `timestamp` column, while potentially useful for temporal analysis, is not immediately relevant.  Therefore, we'll drop this column to reduce the dataset's dimensionality and computational burden.

3.  **Duplicate Elimination:** We'll rigorously identify and remove duplicate entries within the merged DataFrame, ensuring each user-movie rating pair is unique. This step is crucial to prevent biases in subsequent analyses and models arising from duplicate ratings.

4.  **Genre Transformation:** The `genres` column in the `movies` DataFrame contains multiple genres per movie separated by '|'. We'll split this column into a list of individual genres to enable genre-based analysis and filtering, as well as potential integration into content-based filtering strategies if desired later.

5. **Title-Year Separation:** We will also extract the release year from the movie title and store it as a separate column for further analysis.
 
6. **Bayesian Average Calculation:** We'll compute the Bayesian Average for each movie's rating.  This approach mitigates the inherent bias of simple average ratings, especially for movies with few ratings, by incorporating the global average rating and the average number of votes per movie. The Bayesian Average formula is:

```
Bayesian Average = ( (C * m) + (R * v) ) / (C + v)
```

where:

*   R = average rating for the movie
*   v = number of votes for the movie
*   m = average rating across all movies
*   C = average number of votes per movie across all movies

By comparing the Bayesian Average with simple average ratings, we'll assess the impact of this adjustment on the perceived popularity of movies, particularly those with limited ratings. We'll visually represent this comparison to discern how the Bayesian Average smooths out the rating distribution.

###EDA

## Univariate Analysis
### Top and Bottom Rated Movies & Users:

Top Rated Movies:
| title                              | avg_rating   |
|:-----------------------------------|:-------------|
| Shawshank Redemption, The          | 4.30266      |
| Fight Club                         | 4.12902      |
| Godfather, The                     | 4.12635      |
| Star Wars: Episode IV - A New Hope | 4.10989      |
| Pulp Fiction                       | 4.09966      |

Bottom Rated Movies:
| title                           | avg_rating   |
|:--------------------------------|:-------------|
| Wild Wild West                  | 2.83571      |
| Coneheads                       | 2.89892      |
| Batman & Robin                  | 2.91389      |
| Anaconda                        | 2.94906      |
| I Know What You Did Last Summer | 2.95827      |

Top Raters (By Average Rating):
| userId   | avg_rating   |
|:---------|:-------------|
| 549      | 3.94902      |
...
| 147      | 20           |
| 189      | 20           |
| 194      | 20           |
| 207      | 20           |
Output is truncated. 

### Bivariate Analysis

The MovieLensBivariateEDA class below performs bivariate analysis on the MovieLens dataset. It investigates the relationship between the average movie rating and two other variables: year of release and genre. This helps us understand how ratings vary depending on when a movie was released and what genre it belongs to. 

analyze_year_vs_avg_rating() method calculates and visualizes the average rating of movies for each year using a line plot. It reveals trends in how movie ratings have evolved over time while analyze_genres_vs_avg_rating() method creates a new DataFrame with individual rows for each movie-genre combination along with their average ratings. It then uses a strip plot to visualize the distribution of average ratings across different genres, helping to identify if certain genres tend to receive higher or lower ratings on average.

### Multvariate

The `MultivariateAnalysis` class begins by initializing with a file path to the CSV data and the number of latent features for dimensionality reduction, which is set via `n_components`. It reads the data from the specified file using Pandas and prepares it for analysis. The `_prepare_data` method maps each movie ID to its title and index, then formats the data for the Surprise library, which is used to train the SVD model. The SVD model, initialized with the specified number of latent features, is trained on the prepared dataset. For similarity calculations, the `find_similar_movies` method computes the cosine similarity between the target movie vector and all other movie vectors. It then retrieves the top `k` most similar movies, excluding the target movie itself. The `plot_similarity_heatmap` method visualizes the similarity between the target movie and its top `k` similar movies by creating a heatmap, with movie titles as labels and similarity scores as annotations. 


## Modeling

The `CollaborativeFiltering` class is designed to recommend movies similar to a given movie based on collaborative filtering techniques. Upon initialization, it stores the user-item matrix \( \mathbf{X} \), movie titles, and various mappings for users and movies. The `find_similar_movies` method identifies similar movies using the specified distance metric, either 'cosine' or 'euclidean'. It transposes the user-item matrix \( \mathbf{X} \) and checks if the movie ID exists in the dataset. If valid, it retrieves the movie vector and applies the `NearestNeighbors` algorithm to find the closest neighbors. The cosine similarity or Euclidean distance is computed as:

The `HybridRecommender` class integrates content-based and collaborative filtering approaches to provide movie recommendations. It initializes with a user-item matrix \( \mathbf{X} \), a genre matrix, and various mappings for users and movies, along with thresholds for user and movie ratings. The `movie_finder` method uses fuzzy matching to find the closest movie title to a given input. The `get_content_based_recommendations` method computes similarity scores between the target movie's genre vector and all other genre vectors using cosine similarity:

The `hybrid_recommendations` method decides whether to use content-based or collaborative filtering based on the user's rating history or the movie's rating history. It applies filtering based on thresholds for user and movie ratings to choose the most appropriate recommendation approach.

The hybrid recommendation system has successfully provided a diverse set of movie suggestions based on user history and content-based filtering. The recommendations, demonstrate the model's capability to blend collaborative and content-based approaches effectively. By integrating user preferences with movie attributes, the system enhances the relevance of suggested titles. This method not only addresses the cold start problem but also ensures that the recommendations cater to various user tastes.


Deployment

* SVD App - https://movielenssvd.streamlit.app/
* Hybrid Recommender App - https://movielenshybridrecommender.streamlit.app/

### Conclusion

In this project, we developed and evaluated several recommendation models based on the MovieLens dataset. We implemented collaborative filtering and hybrid recommendation systems to address various aspects of movie recommendation. The Singular Value Decomposition (SVD) model and the Hybrid Recommender were deployed using Streamlit, providing a user-friendly interface for generating and visualizing movie recommendations. However, the evaluation of these models was not performed in detail due to limitations in computational power and potential biases introduced by the data splitting process. Specifically, the split did not fully capture the variability in user ratings or movie preferences, leading to skewed performance metrics. The true effectiveness of the recommendations could be more accurately assessed through continuous data collection and analysis of user feedback on the recommendations over time.

### Recommendations

1. **Enhance Data Collection**: Continuously gather user feedback and ratings to better assess the performance and relevance of the recommendations. This approach will help in understanding user preferences and improving model accuracy.

2. **Explore Advanced Metrics**: Consider integrating advanced evaluation metrics such as Mean Average Precision (MAP) and Normalized Discounted Cumulative Gain (NDCG) to gain deeper insights into model performance beyond traditional regression metrics like RMSE and MAE.

3. **Optimize Computational Resources**: Evaluate the models on a more extensive scale by using cloud computing resources or optimizing the current computational setup to handle larger datasets and more complex models.

### Next Steps

1. **User Feedback Integration**: Develop a mechanism to collect real-time user feedback on recommendations. This data will provide insights into model performance and areas for improvement.

2. **Model Enhancement**: Experiment with additional recommendation algorithms and fine-tune existing models based on user feedback and performance metrics.

3. **Deployment Improvements**: Enhance the Streamlit applications for better user experience and incorporate features for dynamic updates based on user interactions and new data.

4. **Scalability and Performance**: Explore methods to scale the recommendation system to handle larger datasets efficiently and improve computational performance.

5. **Cross-Validation**: Implement cross-validation techniques to ensure robust model evaluation and mitigate potential biases in the training and testing processes.
