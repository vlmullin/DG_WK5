
import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate


# Load data into pandas dataframes
movies_df = pd.read_csv('https://raw.githubusercontent.com/vlmullin/DG_WK4/main/small/movies.csv')
ratings_df = pd.read_csv('https://raw.githubusercontent.com/vlmullin/DG_WK4/main/small/ratings.csv')

# Define the genre options
genre_options = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

# Define the SVD model
reader = Reader()
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Get all unique user IDs and unrated movies
user_ids = ratings_df['userId'].unique()
#unrated_movies = movies_df.loc[~movies_df['movieId'].isin(ratings_df['movieId'])]

# Use the SVD model to predict the ratings for all user ID and unrated movie combinations
predictions = []
for user_id in user_ids:
    for movie_id in movies_df['movieId']:
        actual_rating = ratings_df.loc[(ratings_df['userId'] == user_id) & (ratings_df['movieId'] == movie_id), 'rating'].values
        if len(actual_rating) > 0:
            actual_rating = actual_rating[0]
            pred_rating= None
        else:
            actual_rating = None
            pred_rating = svd.predict(user_id, movie_id).est
        genre=movies_df[(movies_df['movieId']==movie_id)]['genres']
        genre=genre.values
        genre[0]
        predictions.append((user_id, movie_id, actual_rating, pred_rating, genre))

# Convert the predictions to a pandas DataFrame and save it to a new CSV file
predictions_df = pd.DataFrame(predictions, columns=['user_id', 'movie_id', 'rating', 'pred_rating', 'genre'])
predictions_df.to_csv('predicted_ratings.csv', index=False)
