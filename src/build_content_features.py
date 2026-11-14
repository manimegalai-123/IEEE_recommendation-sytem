
import pandas as pd
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MOVIES_FILE = BASE_DIR / "dataset" / "processed" / "movie_mapping.csv"
OUTPUT_DIR = BASE_DIR / "dataset" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load movie metadata
movies = pd.read_csv(MOVIES_FILE)

print("Loaded movies:", len(movies))

# Prepare text features
movies["combined_features"] = (
    movies["combined_features"]
    .fillna("")
    .astype(str)
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(
    movies["combined_features"]
)

print("\nTF-IDF matrix created!")
print("Matrix shape:", tfidf_matrix.shape)

# Save TF-IDF matrix
save_npz(
    OUTPUT_DIR / "movie_tfidf.npz",
    tfidf_matrix
)

# Save trained vectorizer
joblib.dump(
    vectorizer,
    OUTPUT_DIR / "tfidf_vectorizer.joblib"
)

# Save movie metadata in the exact same row order
movies[
    ["movieId", "title", "genres"]
].to_csv(
    OUTPUT_DIR / "content_movie_index.csv",
    index=False
)

print("\nContent features saved successfully!")
print("Files created:")
print("- movie_tfidf.npz")
print("- tfidf_vectorizer.joblib")
print("- content_movie_index.csv")
