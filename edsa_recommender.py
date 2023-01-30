"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
#from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
#title_list = load_movie_titles('resources/data/movies.csv')
model=pickle.load(open('resources/models/SVD.pkl', 'rb'))
movies_df = pd.read_csv('resources/data/movies.csv', sep = ',')
movies_df.dropna()
ratings_df = pd.read_csv('resources/data/ratings_lit.csv')
ratings_df = ratings_df.drop(['timestamp'], axis=1)
title_list = movies_df['title'].to_list()
# load additional files
#imdb_df = pd.read_csv('resources/data/imdb_data.csv')
# App declaration


def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System", "Solution Overview",
                    "Model Predictions", "Model Comparison", "Project Description", "Team"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png', use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option', title_list[14930:15200])
        movie_2 = st.selectbox('Second Option', title_list[25055:25255])
        movie_3 = st.selectbox('Third Option', title_list[21100:21200])
        fav_movies = [movie_1, movie_2, movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i, j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")

        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i, j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")

    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("Describe your winning approach on this page")
    if page_selection == "Model Predictions":
        st.info("Prediction with ML Models")
        st.write("The movies below are recommended based on the collaborative model which for this exercise we chose the SVD model")
        st.image("resources/imgs/model_image.jpeg")
        st.write("Please choose your 3 favourite movies from the table below...")

        fav_movies = st.multiselect(
            'Select favourite movies:', title_list[14930:15200], max_selections=3)
        st.write('### Selected Movies')
        st.table(pd.DataFrame(
            fav_movies, columns=["Title"]))

        st.write("Please click the button below to see the recommended movies")
        if st.button("Recommend"):
            try:
                with st.spinner('Crunching the numbers...'):
                    top_recommendations = collab_model(movie_list=fav_movies,
                                                       top_n=10)
                st.title("We think you'll like:")
                for i, j in enumerate(top_recommendations):
                    st.subheader(str(i+1)+'. '+j)
            except:
                st.error("Oops! Looks like this algorithm does't work.\
                            We'll need to fix it!")
    if page_selection == "Model Comparison":
        st.info(
            "This page shows a comparison of the model performance")
    if page_selection == "Project Description":
        st.info("More Information about the project")
        st.image("resources/imgs/dc_marvel.webp")
        st.markdown("""
        In today’s technology driven world, recommender systems are socially and 
        economically critical for ensuring that individuals can make appropriate 
        choices surrounding the content they engage with on a daily basis. One 
        application where this is especially true surrounds movie content recommendations; 
        where intelligent algorithms can help viewers find great titles from tens of 
        thousands of options.""")

        st.markdown("""...ever wondered how Netflix, Amazon Prime, Showmax, Disney and the likes somehow 
        know what to recommend to you?
        \n ...it's not just a guess drawn out of the hat. There is an algorithm behind it.
        \n With this context, EDSA is challenging you to construct a recommendation algorithm 
        based on content or collaborative filtering, capable of accurately predicting how a 
        user will rate a movie they have not yet viewed based on their historical preferences.
        \n What value is achieved through building a functional recommender system?
        Providing an accurate and robust solution to this challenge has immense economic potential, 
        with users of the system being exposed to content they would like to view or purchase - 
        generating revenue and platform affinity.""")

    if page_selection == "About Us":
        st.info(
            "Below is detailed information about the our business and the team")
    if page_selection == "Team":
        st.info(
            "Below is information about the team")
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Chima", "Kazeem", "Thepe", "Dotun"])

        with tab1:
            st.subheader("Chima Enyeribe")
            st.markdown("Data Scientist")
            st.image("resources/imgs/Jasper.jpg", width=200)
            st.markdown("jasperobed@gmail.com")
        with tab2:
            st.subheader("Kazeem Okunola")
            st.markdown("Chief Financial Officer")
            st.image("resources/imgs/kazeem.webp", width=200)
            st.markdown("achieverk2@gmail.com")
        with tab3:
            st.subheader("Thepe Mashala")
            st.markdown("Chief Operations Officer")
            st.image("resources/imgs/8324.jpg", width=200)
            st.markdown("tpmashala@gmail.com")
        with tab4:
            st.subheader("Dotun Olasogba")
            st.markdown("Chief Digital Officer")
            st.image("resources/imgs/dotun.jpeg", width=200)
            st.markdown("dotunolasogba@yahoo.com")
        # You may want to add more sections here for aspects such as an EDA,
        # or to provide your business pitch.


if __name__ == '__main__':
    main()
