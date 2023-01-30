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
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')
# load additional files
# imdb_df = pd.read_csv('resources/data/imdb_data.csv')
# App declaration


def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Company Profile", "About the Project", "Solution Overview", "Recommender System",
                    "Model Comparison", "Team"]

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

    if page_selection == "Company Profile":
        st.title("Company Profile")
        st.image("resources/imgs/dc_marvel.webp")
        st.write("""
        TriTech Inc. was established in April 2015 to provide digital technology solutions for a wide range of domain businesses. Our founding mission was to cultivate professional relationships with our customers to provide reliable digital technology solutions for their requirements. """)
        st.write("\n")
        st.write("""The team at TriTech Inc. is equipped with a wide range of cross-domain skillset developed over years of experience not only in information technology but also in business processes, financial services, construction and telecommunications across a range of various subdomains.
        This business experience makes service offerings uniquely positioned to provide solutions promising better cost savings, operational efficiency and productivity gains for each of our customers, regardless of their industry.
        """)
        st.write("\n")
        st.write("""As a well established technology-agnostic company, we pride ourselves on providing a detailed suite of futuristic solutions. These comprise of digital automation consultancy, technology transformation services, custom AI solutions and Machine Learning software as well as enterprise architecture establishment.
        Our team consistently delivers state-of-the-art services across different domains including, integrated digital business solutions, AI & ML applications, IOT and digital transformation and/or management services.
        """)
        st.write("\n")
        st.write("""At TriTech Inc., we guarantee rapid, reliable and robust turn-key digital solutions that will transform your business.""")
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("Describe your winning approach on this page")
    # if page_selection == "Model Predictions":
    #     st.info("Prediction with ML Models")
    #     st.write("The movies below are recommended based on the collaborative model which for this exercise we chose the SVD model")
    #     st.image("resources/imgs/model_image.jpeg")
    #     st.write("Please choose your 3 favourite movies from the table below...")

    #     fav_movies = st.multiselect(
    #         'Select favourite movies:', title_list[14930:15200], max_selections=3)
    #     st.write('### Selected Movies')
    #     st.table(pd.DataFrame(
    #         fav_movies, columns=["Title"]))

    #     st.write("Please click the button below to see the recommended movies")
    #     if st.button("Recommend"):
    #         try:
    #             with st.spinner('Crunching the numbers...'):
    #                 top_recommendations = collab_model(movie_list=fav_movies,
    #                                                    top_n=10)
    #             st.title("We think you'll like:")
    #             for i, j in enumerate(top_recommendations):
    #                 st.subheader(str(i+1)+'. '+j)
    #         except:
    #             st.error("Oops! Looks like this algorithm does't work.\
    #                         We'll need to fix it!")
    if page_selection == "Model Comparison":
        st.info(
            "This page shows a comparison of the model performance")
    if page_selection == "About the Project":
        st.title("About the project")
        st.image("resources/imgs/dc_marvel.webp")
        st.write("""
        **The main goal of this project is to predict movie ratings using the
        MovieLens(10M) dataset, which contains the ratings of several movies
        given by various users**""")
        st.write("\n")
        st.markdown("""
        **The problem context for this project is as follows:** \n
        Traditional movie recommendation systems often rely on a fixed set of rules or
        pre-determined criteria, which may not take into account the individual preferences
        of the users. This can lead to less accurate or relevant recommendations, which can
        negatively impact customer engagement and ultimately result in lost revenue for the company.""")
        st.markdown("""By developing a more personalized and accurate recommendation system, the project aims to improve customer engagement and increase revenue for the company.
        \n **The objectives of this project are as follows:** \n""")
        st.write("""+ To analyze the provided movie data and identify patterns and trends that can be used to generate more accurate and personalized movie recommendations for users.""")
        st.write("""+ To develop a recommendation system that takes into account the individual preferences of the users and generates personalized movie recommendations based on the data provided by the company.""")
        st.write("""+ To evaluate the performance of the developed recommendation system and make improvements as necessary to ensure that it is accurate and effective in generating relevant recommendations for users.""")
        st.write("""+ To make recommendations for the company to improve the effectiveness of their movie recommendations and increase customer engagement and revenue.""")

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
            st.markdown("Chief Executive Officer")
            st.image("resources/imgs/chima.webp", width=200)
            st.write("""
            Chima Enyeribe is the founder and CEO of TriTech Inc., a position he has held since the company's inception in April 2015. Prior to founding TriTech, Chima worked for Zen Bank in Nigeria, providing financial service solutions to individuals and corporates across 19 countries in Africa, Latin America and Asia.
            Under his leadership, the Bank serviced over 5 million customers and had $3 million of assets under management.
            Chima is a seasoned business leader with experience in engineering, financial services and construction. Since founding TriTech, Chima has played a pivotal role in the strengthening the company's financial position, strategy formulation and successful listing on the Nigerian and Ghanaian Stock Exchanges.
            """)
            st.markdown("jasperobed@gmail.com")
        with tab2:
            st.subheader("Kazeem Okunola")
            st.markdown("Chief Financial Officer")
            st.image("resources/imgs/kazeem.webp", width=200)
            st.write("""
            Kazeem is a co-founder and Chief Financial Officer of TriTech Inc. Kazeem previously worked at Central Bank of Nigeria where he held the positions of Deputy Chief Financial Officer and Executive Risk and Compliance Officer, prior to him co-founding TriTech Inc. and becoming the Chief Financial officer in 2015.
            Before working at the Central Bank of Nigeria, Kazeem was employed as a Finance Director of MSC Telco Holdings, and an executive director of the company's main board. He has also held various other roles at MSC Telco Holdings.
            In addition to his, extensive financial industry experience, Kazeem has also worked in the consulting and services. He brings with him extensive financial strategy, financial management and accounting experience, with a strong background in enterprise risk management and financial controls.
            """)
            st.markdown("achieverk2@gmail.com")
        with tab3:
            st.subheader("Thepe Mashala")
            st.markdown("Chief Operations Officer")
            st.image("resources/imgs/8324.jpg", width=200)
            st.write("""
            Thepe Mashala joined TriTech Inc. in March 2018. Prior to his appointment as the Chief Operations Officer, he led the establishment of Blue Sky Group's strategic M&A business, the launch of the Group's flagship Advanced Data Analytics platform, and
            helped shape the Group's strategic direction across a range of key topics.
            Before joining TriTech Inc., Thepe led the Southern Africa Digital Practice for McKinsey & Company, where he focused on Digital Business Building and operational transformations for telecommunications businesses, financial services, and SMME sector clients.
            """)
            st.markdown("tpmashala@gmail.com")
        with tab4:
            st.subheader("Dotun Olasogba")
            st.markdown("Chief Digital Officer")
            st.image("resources/imgs/dotun.jpeg", width=200)
            st.write("""
            Dotun joined TriTech Inc. in 2020 from MMB, where he led the debt-based financing business and was responsible for financing solutions, digital transformation and M&A execution.
            Dotun's experience and leadership capacity is important to the company's digital portfolio optimisation drive, itself key to the strategy. This includes the digital transformation
            realisation programme, which aims to reduce legacy systems, simplify the digital operations portfolio, reduce risk, improve digital footprint and take advantage of expansion opportunities.
            """)
            st.markdown("dotunolasogba@yahoo.com")
        # You may want to add more sections here for aspects such as an EDA,
        # or to provide your business pitch.


if __name__ == '__main__':
    main()
