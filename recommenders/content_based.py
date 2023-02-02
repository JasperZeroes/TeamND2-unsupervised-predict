"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
<<<<<<< HEAD
movies = pd.read_csv('resources/data/movies.csv', sep = ',')
ratings = pd.read_csv('resources/data/ratings.csv')
movies.dropna(inplace=True)
=======
#movies = pd.read_csv('resources/data/movies_lit.csv', sep = ',')
#ratings_df = pd.read_csv('resources/data/ratings_lit.csv')
#ratings_df = ratings_df.drop(['timestamp'], axis=1,inplace=True)
# movies.dropna(inplace=True)

>>>>>>> 61ac149c27c6ebfbf0d029b1653e11b7e072cbe9

def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.
    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.
    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.
    """
<<<<<<< HEAD
    # Split genre data into individual words.
    movies['keyWords'] = movies['genres'].str.replace('|', ' ')
    # Subset of the data
    movies_subset = movies[:subset_size]
=======
    # Subset of the data
    movies_subset = movies.loc[:subset_size]

    # Split genre data into individual words.
    movies_subset["keyWords"] = movies_subset["genres"].str.replace("|", " ")
    movies_subset["keyWords"] = movies_subset["keyWords"].str.lower()

>>>>>>> 61ac149c27c6ebfbf0d029b1653e11b7e072cbe9
    return movies_subset

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.
def content_model(movie_list, top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.
    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.
    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.
    """
<<<<<<< HEAD
    # Initializing the empty list of recommended movies
    recommended_movies = []
    data = data_preprocessing(27000)
=======
    # process a subset of the dataframe
    data = data_preprocessing(27000)

>>>>>>> 61ac149c27c6ebfbf0d029b1653e11b7e072cbe9
    # Instantiating and generating the count matrix
    count_vec = CountVectorizer()
    count_matrix = count_vec.fit_transform(data['keyWords'])
    indices = pd.Series(data['title'])
<<<<<<< HEAD
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    # Getting the index of the movie that matches the title
    idx_1 = indices[indices == movie_list[0]].index[0]
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]
    # Creating a Series with the similarity scores in descending order
    rank_1 = cosine_sim[idx_1]
    rank_2 = cosine_sim[idx_2]
    rank_3 = cosine_sim[idx_3]
    # Calculating the scores
    score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
    score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
    score_series_3 = pd.Series(rank_3).sort_values(ascending = False)
    # Getting the indexes of the 10 most similar movies
    listings = score_series_1.append(score_series_1).append(score_series_3).sort_values(ascending = False)
=======

    # Getting the index of the movies that match the title
    idx_list = [indices[indices == movie].index[0] for movie in movie_list]

    # Creating a Series with the similarity scores in descending order
    score_series_list = [pd.Series(cosine_sim[idx]).sort_values(
        ascending=False) for idx in idx_list]

    # Appending the names of movies
    listings = pd.concat(score_series_list).sort_values(ascending=False)
>>>>>>> 61ac149c27c6ebfbf0d029b1653e11b7e072cbe9

    # Store movie names
    recommended_movies = []
<<<<<<< HEAD
=======

>>>>>>> 61ac149c27c6ebfbf0d029b1653e11b7e072cbe9
    # Appending the names of movies
    top_50_indexes = list(listings.iloc[1:50].index)
    # Removing chosen movies
<<<<<<< HEAD
    top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    for i in top_indexes[:top_n]:
        recommended_movies.append(list(movies['title'])[i])
=======
    top_indexes = np.setdiff1d(top_50_indexes, idx_list)

    for i in top_indexes[:top_n]:
        recommended_movies.append(list(data['title'])[i])
>>>>>>> 61ac149c27c6ebfbf0d029b1653e11b7e072cbe9
    return recommended_movies
