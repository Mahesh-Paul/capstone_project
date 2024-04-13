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

def fetch_movie_thumbnail(movie_id):
    api_key = "c7ec19ffdd3279641fb606d19ceb9bb1" 
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    
    try:
        response = requests.get(url)
        data = response.json()
        poster_path = data['poster_path']
        full_path = f"https://image.tmdb.org/t/p/w185/{poster_path}" 
        return full_path
    except Exception as e:
        print(f"Error fetching movie thumbnail: {e}")
        return None

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

    total_reviews = len(filtered_reviews)
    positive_count = len(positive_reviews)
    negative_count = len(negative_reviews)
    neutral_count = len(neutral_reviews)

    sentiment_stats = f"Reviews:\n" \
                      f"Positive: {positive_count}\n" \
                      f"Negative: {negative_count}\n" \
                      f"Neutral: {neutral_count}"

    return sentiment_stats

def recommend_books(book_title, books_df, tfidf_matrix, cosine_sim):
    idx = books_df[books_df['title'].str.contains(book_title, case=False)].index
    if len(idx) == 0:
        st.error(f"No book found with the title '{book_title}'.")
        return pd.DataFrame() 
    else:
        idx = idx[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]  
        recommended_books = books_df.iloc[[i[0] for i in sim_scores]]
        return recommended_books

# Function to fetch book summary from text file
def fetch_book_summary(book_title):
    # Path to your booksummaries.txt file
    file_path = "booksummaries.txt"

    # Open the text file and search for the book title
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # Split each line into fields using a separator (e.g., tab or comma)
            fields = line.strip().split("\t")  # Assuming tab-separated format, adjust as needed
            # Check if the first field matches the given book title
            if fields[2].lower() == book_title.lower():  # Assuming title is in the third field, adjust as needed
                return fields[6]  # Assuming summary is in the seventh field, adjust as needed
    # If book title is not found, return None
    return None

# Function to preprocess text
def preprocess_text(text):
    if isinstance(text, str):
        return ' '.join(text.lower().split())
    else:
        return ''

# Function to recommend songs
def recommend_songs(song_title, songs_df, tfidf_vectorizer, tfidf_matrix):
    song_title = preprocess_text(song_title)
    song_vector = tfidf_vectorizer.transform([song_title])
    cosine_sim = linear_kernel(song_vector, tfidf_matrix)
    similar_indices = cosine_sim.argsort().flatten()[::-1]
    recommended_songs = songs_df.iloc[similar_indices][:5]
    return recommended_songs

def fetch_song_thumbnail(song_id):
    return None

# Streamlit app
def main():
    st.title("Recommendation System")

    option = st.sidebar.selectbox("Select recommendation type", ["Movie", "Book", "Song"])

    if option == "Movie":
        movie_name = st.text_input("Enter a movie name:")
        show_movie_recommendations = st.button("Show Movie Recommendations")
        
        if movie_name and show_movie_recommendations:
            # Load movie data
            movies_df = load_movie_data()

            # Create TF-IDF vectorizer
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')

            # Construct TF-IDF matrix
            tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['title'].fillna(''))

            # Compute cosine similarity matrix
            cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

            # Get movie recommendations
            recommended_movies = recommend_movies(movie_name, movies_df, tfidf_matrix, cosine_sim)

            # Import IMDB reviews dataset
            imdb_reviews = import_imdb_reviews("imdb_reviews.csv")

            # Display movie recommendations
            st.write("Movie Recommendations:")
            for i, row in recommended_movies.iterrows():
                st.write(f"Title: {row['title']}")
                # Display movie poster
                poster_url = fetch_movie_thumbnail(row['id'])
                if poster_url:
                    try:
                        response = requests.get(poster_url)
                        img = Image.open(BytesIO(response.content))
                        img = img.resize((100, 100))  # Resize image to 100x100 pixels
                        st.image(img, caption="Movie Poster", use_column_width=False)
                    except Exception as e:
                        st.error(f"Error loading poster for {row['title']}: {e}")
                else:
                    st.write("Poster not available")

                # Analyze movie sentiment
                if imdb_reviews is not None:
                    sentiment_stats = analyze_movie_sentiment(imdb_reviews, row['title'])
                    st.write(sentiment_stats)
                
                st.write("---")

        elif not movie_name:
            st.write("Please enter a movie name.")

    elif option == "Book":
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
                # Display thumbnail image
                thumbnail_url = row['thumbnail']
                if thumbnail_url:
                    try:
                        response = requests.get(thumbnail_url)
                        img = Image.open(BytesIO(response.content))
                        img = img.resize((100, 100))  # Resize image to 100x100 pixels
                        st.image(img, caption="Book Thumbnail", use_column_width=False)
                    except Exception as e:
                        st.error(f"Error loading thumbnail for {row['title']}: {e}")
                else:
                    st.write("Thumbnail not available")
                
                # Add "Review more" button with a unique key
                review_button_key = f"review_button_{i}"
                if st.button("Review more", key=review_button_key):
                    # Fetch and display book summary
                    book_summary = fetch_book_summary(row['title'])
                    if book_summary:
                        st.write(f"Book Summary:")
                        st.write(book_summary)
                    else:
                        st.write("Summary not available")
                    st.write("---")

                st.write("---")

        elif not input_title:
            st.write("Please enter a book title.")

    else:
        # Section for song recommendations
        input_song = st.text_input("Enter a song name:")
        show_song_recommendations = st.button("Show Song Recommendations")
        
        if input_song and show_song_recommendations:
            # Load song data
            songs_df = pd.read_csv("spotify_songs.csv")  # Assuming CSV file contains song data

            # Create TF-IDF vectorizer
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')

            # Construct TF-IDF matrix
            tfidf_matrix = tfidf_vectorizer.fit_transform(songs_df['track_name'].fillna(''))

            # Get song recommendations
            recommended_songs = recommend_songs(input_song, songs_df, tfidf_vectorizer, tfidf_matrix)

            # Display song recommendations
            st.write("Song Recommendations:")
            for i, row in recommended_songs.iterrows():
                st.write(f"Title: {row['track_name']}")
                st.write(f"Artist: {row['track_artist']}")
                st.write(f"Album: {row['track_album_name']}")
                st.write(f"Popularity: {row['track_popularity']}")
                # Display thumbnail image
                thumbnail_url = fetch_song_thumbnail(row['track_id'])  # You need to implement this function
                if thumbnail_url:
                    try:
                        response = requests.get(thumbnail_url)
                        img = Image.open(BytesIO(response.content))
                        img = img.resize((100, 100))  # Resize image to 100x100 pixels
                        st.image(img, caption="Song Thumbnail", use_column_width=False)
                    except Exception as e:
                        st.error(f"Error loading thumbnail for {row['track_name']}: {e}")
                else:
                    st.write("")
                st.write("---")
                st.write("")  # Add an empty line after each set of song recommendations

        elif not input_song:
            st.write("Please enter a song name.")

if __name__ == "__main__":
    main()
