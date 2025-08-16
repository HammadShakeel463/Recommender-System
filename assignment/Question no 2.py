import pandas as pd
import numpy as np

# Load the user ratings dataset (user_id, movie_id, rating, timestamp)
ratings_df = pd.read_csv('ratings.csv')

# Load the movie information dataset (movie_id, title, genre)
movies_df = pd.read_csv('movies.csv')

# Ask the user for their user_id and movie name
user_id = int(input("Enter your user ID: "))
movie_name = input("Enter the movie name you want recommendations for: ")

# Find the movie_id for the given movie name
movie_id = movies_df[movies_df['title'] == movie_name]['movieId'].values[0]


# Calculate mean ratings for each movie
movie_ratings = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
movie_ratings.columns = ['movieId', 'mean_rating', 'rating_count']

# Calculate Adjusted Cosine Similarity between user and all other users
user_ratings = ratings_df[ratings_df['userId'] == user_id]
user_mean_rating = user_ratings['rating'].mean()

# Create a pivot table to compare user ratings with other users
pivot_ratings = ratings_df.pivot_table(index='userId', columns='movieId', values='rating')

# Calculate Adjusted Cosine Similarity for each user
similarity_scores = {}
for user in pivot_ratings.index:
    if user == user_id:
        continue  # Skip the active user
    sim_numerator = 0
    sim_denominator_user = 0
    sim_denominator_curr_user = 0
    for movie in pivot_ratings.columns:
        if not np.isnan(pivot_ratings[movie][user]) and not np.isnan(pivot_ratings[movie][user_id]):
            rating_u = pivot_ratings[movie][user] - movie_ratings[movie_ratings['movieId'] == movie]['mean_rating'].values[0]
            rating_i = pivot_ratings[movie][user_id] - movie_ratings[movie_ratings['movieId'] == movie]['mean_rating'].values[0]
            sim_numerator += rating_u * rating_i
            sim_denominator_user += rating_u ** 2
            sim_denominator_curr_user += rating_i ** 2
    sim_score = sim_numerator / (np.sqrt(sim_denominator_user) * np.sqrt(sim_denominator_curr_user))
    similarity_scores[user] = sim_score

# Find top-3 most similar users
similar_users = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:3]

movie_recommendations = {}
for user, similarity in similar_users:
    user_ratings = pivot_ratings.loc[user]
    for movie in pivot_ratings.columns:
        if np.isnan(pivot_ratings[movie][user_id]) and not np.isnan(user_ratings[movie]):
            if movie not in movie_recommendations:
                movie_recommendations[movie] = 0
            movie_recommendations[movie] += similarity * (user_ratings[movie] - movie_ratings[movie_ratings['movieId'] == movie]['mean_rating'].values[0])

# Sort the movie recommendations by score
movie_recommendations = sorted(movie_recommendations.items(), key=lambda x: x[1], reverse=True)




# Display the top-3 recommended movies to the user
print(f"Top-3 movie recommendations for User {user_id} based on movie '{movie_name}':")
for i, (movie, score) in enumerate(movie_recommendations[:3], 1):
    movie_title = movies_df[movies_df['movieId'] == movie]['title'].values[0]
    print(f"{i}. {movie_title} (Score: {score:.2f})")



