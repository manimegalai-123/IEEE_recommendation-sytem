
import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Input files
MOVIES_FILE = BASE_DIR / "dataset" / "raw" / "ml-25m" / "movies.csv"

# Output directory
OUTPUT_DIR = BASE_DIR / "dataset" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "movie_mapping.csv"

if not MOVIES_FILE.exists():
    raise FileNotFoundError(f"Missing file: {MOVIES_FILE}")

# Load MovieLens movie metadata
movies = pd.read_csv(MOVIES_FILE)

print("Total MovieLens movies:", len(movies))

# Clean movie metadata
movies = movies[["movieId", "title", "genres"]].copy()

movies["title"] = movies["title"].fillna("")
movies["genres"] = movies["genres"].fillna("")

# Create content features
movies["combined_features"] = (
    movies["title"] + " " + movies["genres"].str.replace("|", " ", regex=False)
)

# Remove duplicate movie IDs
movies = movies.drop_duplicates(subset=["movieId"])

# Save mapping
movies.to_csv(OUTPUT_FILE, index=False)

print("\nMovie metadata mapping created successfully!")
print("Mapped movies:", len(movies))
print("Saved to:", OUTPUT_FILE)

print("\nSample records:")
print(movies.head())
