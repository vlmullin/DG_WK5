from flask import Flask, render_template, request
import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

app = Flask(__name__)

# Load data into pandas dataframes
movies_df = pd.read_csv('https://raw.githubusercontent.com/vlmullin/DG_WK4/main/small/movies.csv')
ratings_df = pd.read_csv('https://raw.githubusercontent.com/vlmullin/DG_WK4/main/small/ratings.csv')
pred_ratings_df=pd.read_csv('https://raw.githubusercontent.com/vlmullin/DG_WK4/main/small/predicted_ratings.csv')

# Define the genre options
genre_options = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

# Define the home page
@app.route('/')
def home():
    return render_template('home.html', genre_options=genre_options)

# Define the recommendation page
@app.route('/recommendations', methods=['POST'])
def recommendations():
    # Get the user ID and genre from the form
    user_id = request.form['userId']
    genre = request.form['genre']

    # Check if the user ID is valid
    if not user_id.isdigit() or int(user_id) < 1 or int(user_id) > 610:
        return render_template('error.html', message='Invalid user ID. Please enter a number between 1 and 610.')

    # Check if the genre is valid
    if genre not in genre_options:
        return render_template('error.html', message='Invalid genre. Please select a genre from the dropdown menu.')

    # Filter the data by user ID and genre
    user_ratings = pred_ratings_df.loc[pred_ratings_df['userId'] == int(user_id)]
    genre_movies = user_ratings.loc[user_ratings['genres'].str.split('|').apply(lambda x: genre in x)]
    recommended_movies = genre_movies.sort_values(by='pred_rating', ascending=False)
    recommended_movies=recommended_movies[0:10]
    recommended_movies = pd.merge(recommended_movies, movies_df[['movieId', 'title']], on='movieId')['title']

    return render_template('recommendations.html', user_id=user_id, genre=genre, recommended_movies=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
