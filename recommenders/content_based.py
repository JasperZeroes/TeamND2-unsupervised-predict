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
#movies = pd.read_csv('resources/data/movies_lit.csv', sep = ',')
#ratings_df = pd.read_csv('resources/data/ratings_lit.csv')
#ratings_df = ratings_df.drop(['timestamp'], axis=1,inplace=True)
# movies.dropna(inplace=True)


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
    # Subset of the data
    movies_subset = movies.loc[:subset_size]

    # Split genre data into individual words.
    movies_subset["keyWords"] = movies_subset["genres"].str.replace("|", " ")
    movies_subset["keyWords"] = movies_subset["keyWords"].str.lower()

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
    # process a subset of the dataframe
    data = data_preprocessing(27000)

    # Instantiating and generating the count matrix
    count_vec = CountVectorizer()
    count_matrix = count_vec.fit_transform(data['keyWords'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    indices = pd.Series(data['title'])

    # Getting the index of the movies that match the title
    idx_list = [indices[indices == movie].index[0] for movie in movie_list]

    # Creating a Series with the similarity scores in descending order
    score_series_list = [pd.Series(cosine_sim[idx]).sort_values(
        ascending=False) for idx in idx_list]

    # Appending the names of movies
    listings = pd.concat(score_series_list).sort_values(ascending=False)

    # Create empty list to store movie names
    recommended_movies = []

    # Appending the names of movies
    top_50_indexes = list(listings.iloc[1:50].index)

    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes, idx_list)

    for i in top_indexes[:top_n]:
        recommended_movies.append(list(data['title'])[i])
    return recommended_movies
