import streamlit as st
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def load_movie_data():
    return pd.read_csv("dataset.csv") 

def load_book_data():
    return pd.read_csv("data.csv")

def recommend_movies(movie_name, movies_df, tfidf_matrix, cosine_sim):
    idx = movies_df[movies_df['title'].str.contains(movie_name, case=False)].index
    if len(idx) == 0:
        st.error(f"No movie found with the name '{movie_name}'.")
        return pd.DataFrame()
    else:
        idx = idx[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        recommended_movies = movies_df.iloc[[i[0] for i in sim_scores]]
        return recommended_movies

def recommend_books(title, books_df, tfidf_matrix, cosine_sim):
    idx = books_df[books_df['title'].str.contains(title, case=False)].index
    if len(idx) == 0:
        st.error(f"No book found with the title '{title}'.")
        return pd.DataFrame() 
    else:
        idx = idx[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        recommended_books = books_df.iloc[[i[0] for i in sim_scores]]
        return recommended_books

def import_imdb_reviews(file_path):
    try:
        dataset = pd.read_csv(file_path)
        return dataset
    except Exception as e:
        st.error("Error loading dataset:", str(e))
        return None

def analyze_movie_sentiment(dataset, movie_name):
    filtered_reviews = dataset[dataset['review'].str.contains(movie_name, case=False)]

    positive_reviews = filtered_reviews[filtered_reviews['sentiment'] == 'positive']
    negative_reviews = filtered_reviews[filtered_reviews['sentiment'] == 'negative']
    neutral_reviews = filtered_reviews[filtered_reviews['sentiment'] == 'neutral']

    st.write(f"Total reviews for '{movie_name}': {len(filtered_reviews)}")
    st.write(f"Positive reviews: {len(positive_reviews)}")
    st.write(f"Negative reviews: {len(negative_reviews)}")
    st.write(f"Neutral reviews: {len(neutral_reviews)}")

def main():
    st.title("Recommendation System")

    option = st.sidebar.selectbox("Select recommendation type", ["Movie", "Book"])

    if option == "Movie":
        movie_name = st.text_input("Enter a movie name:")
        show_movie_recommendations = st.button("Show Movie Recommendations")
        
        if movie_name and show_movie_recommendations:
            movies_df = load_movie_data()

            tfidf_vectorizer = TfidfVectorizer(stop_words='english')

            tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['overview'].fillna(''))

            cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

            recommended_movies = recommend_movies(movie_name, movies_df, tfidf_matrix, cosine_sim)

            st.write("Movie Recommendations:")
            for i, row in recommended_movies.iterrows():
                st.write(f"Title: {row['title']}")
                st.write(f"Genres: {row['genre']}")
                st.write(f"Overview: {row['overview']}")
                st.write("---")

        elif not movie_name:
            st.write("Please enter a movie name.")

    else:
        input_title = st.text_input("Enter a book title:")
        show_recommendations = st.button("Show Book Recommendations")
        
        if input_title and show_recommendations:
            # Load book data
            books_df = load_book_data()

            # Create TF-IDF vectorizer
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')

            # Construct TF-IDF matrix
            tfidf_matrix = tfidf_vectorizer.fit_transform(books_df['description'].fillna(''))

            # Compute cosine similarity matrix
            cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

            # Get book recommendations
            recommended_books = recommend_books(input_title, books_df, tfidf_matrix, cosine_sim)

            # Display book recommendations
            st.write("Book Recommendations:")
            for i, row in recommended_books.iterrows():
                st.write(f"Title: {row['title']}")
                st.write(f"Authors: {row['authors']}")
                st.write(f"Average Rating: {row['average_rating']}")
                st.write("---")

        elif not input_title:
            st.write("Please enter a book title.")

if __name__ == "__main__":
    main()
