
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# 1. Load cleaned dataset
df = pd.read_csv("dataset/movies_cleaned.csv")

# 2. Handle empty combined features
df["combined_features"] = (
    df["combined_features"].fillna("")
)

# 3. TF-IDF vectorization
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

tfidf_matrix = tfidf.fit_transform(
    df["combined_features"]
)

print("TF-IDF matrix shape:", tfidf_matrix.shape)

# 4. Build movie title index
df["title"] = df["title"].astype(str)

# Use normalized titles for searching
df["title_lower"] = df["title"].str.lower().str.strip()

# 5. Recommendation function
def recommend_movies(movie_title, top_n=10):

    movie_title = movie_title.lower().strip()

    matches = df.index[
        df["title_lower"] == movie_title
    ].tolist()

    if not matches:
        print("Movie not found:", movie_title)
        return

    movie_index = matches[0]

    # Calculate similarity against all movies
    similarity_scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    # Sort by highest similarity
    similar_indices = similarity_scores.argsort()[::-1]

    # Exclude input movie
    similar_indices = [
        idx for idx in similar_indices
        if idx != movie_index
    ]

    # Display Top N movies
    recommendations = df.iloc[
    similar_indices[:top_n]
][["title", "genres", "vote_average"]].copy()

    recommendations["genres"] = (
    recommendations["genres"].fillna("Unknown")
)
    return recommendations


# 6. Test the model
if __name__ == "__main__":

    movie = "Toy Story"

    print("\nRecommendations for:", movie)

    results = recommend_movies(movie, top_n=10)

    if results is not None:
        print(results.to_string(index=False))
