
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.sparse import load_npz
from sklearn.preprocessing import normalize


# ==================================================
# 1. PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "dataset/processed/ml25m_subset.pkl"
TFIDF_FILE = BASE_DIR / "dataset/processed/movie_tfidf.npz"
MOVIE_FILE = BASE_DIR / "dataset/processed/content_movie_index.csv"
MODEL_FILE = BASE_DIR / "results/baseline_mf.pt"
HYBRID_FILE = BASE_DIR / "results/hybrid_recommender.json"

device = torch.device("cpu")


# ==================================================
# 2. LOAD DATA
# ==================================================

print("\n========== LOADING RECOMMENDER SYSTEM ==========")

with open(DATA_FILE, "rb") as f:
    D = pickle.load(f)

train = D["train"]
n_users = D["n_users"]
n_items = D["n_items"]

# Load original MovieLens user IDs.
user_map = D["user_map"]

if isinstance(user_map, dict):
    original_user_ids = [
        int(user_map[i] if i in user_map else user_map[str(i)])
        for i in range(n_users)
    ]
else:
    original_user_ids = [int(x) for x in user_map]

# Original MovieLens user ID -> internal user index.
user_to_index = {
    original_id: index
    for index, original_id in enumerate(original_user_ids)
}

# Load movie metadata.
movies = pd.read_csv(MOVIE_FILE)

# Load TF-IDF features.
tfidf_all = load_npz(TFIDF_FILE).tocsr()

# Load selected hybrid alpha.
import json

with open(HYBRID_FILE, "r") as f:
    hybrid_results = json.load(f)

alpha = float(hybrid_results["alpha"])

print("Users:", n_users)
print("Items:", n_items)
print("Hybrid alpha:", alpha)


# ==================================================
# 3. LOAD TRAINED BPR MODEL
# ==================================================

try:
    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=False
    )
except TypeError:
    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device
    )

user_embeddings = checkpoint[
    "model_state_dict"
]["U.weight"]

item_embeddings = checkpoint[
    "model_state_dict"
]["I.weight"]

user_embeddings = user_embeddings.detach().numpy()
item_embeddings = item_embeddings.detach().numpy()


# ==================================================
# 4. MAP INTERNAL ITEM IDS TO MOVIE METADATA
# ==================================================

item_map = D["item_map"]

if isinstance(item_map, dict):
    original_movie_ids = [
        int(item_map[i] if i in item_map else item_map[str(i)])
        for i in range(n_items)
    ]
else:
    original_movie_ids = [int(x) for x in item_map]

movie_to_row = {
    int(movie_id): row
    for row, movie_id in enumerate(movies["movieId"])
}

feature_rows = []

for movie_id in original_movie_ids:
    row = movie_to_row.get(movie_id)

    if row is None:
        raise ValueError(
            f"MovieLens movie ID {movie_id} has no TF-IDF mapping."
        )

    feature_rows.append(row)

# Metadata in exactly the same order as BPR item embeddings.
aligned_movies = (
    movies.iloc[feature_rows]
    .reset_index(drop=True)
)

item_features = tfidf_all[feature_rows].tocsr()


# ==================================================
# 5. CREATE USER CONTENT PROFILES
# ==================================================

rows = train[:, 0]
cols = train[:, 1]

from scipy.sparse import csr_matrix

seen_train = csr_matrix(
    (
        np.ones(len(train), dtype=np.float32),
        (rows, cols)
    ),
    shape=(n_users, n_items)
)

user_profiles = seen_train @ item_features

user_profiles = normalize(
    user_profiles,
    norm="l2",
    axis=1
).tocsr()


# ==================================================
# 6. HYBRID RECOMMENDATION FUNCTION
# ==================================================

def recommend_movies(original_user_id, top_k=10):

    if original_user_id not in user_to_index:
        print(
            "\nUser ID not found in the processed dataset."
        )
        print("Please enter a valid original MovieLens user ID.")
        return

    user_id = user_to_index[original_user_id]

    print("\n========== USER INFORMATION ==========")
    print("Original MovieLens User ID:", original_user_id)
    print("Internal User Index:", user_id)

    # ----------------------------------------------
    # Previously liked movies
    # ----------------------------------------------

    history = train[train[:, 0] == user_id]

    history_movies = aligned_movies.iloc[
        history[:, 1]
    ]

    print("\nPreviously liked movies:")
    print(
        history_movies[
            ["title", "genres"]
        ].head(10).to_string(index=False)
    )

    # ----------------------------------------------
    # Collaborative filtering scores
    # ----------------------------------------------

    cf_scores = (
        user_embeddings[user_id]
        @ item_embeddings.T
    )

    # ----------------------------------------------
    # Content-based scores
    # ----------------------------------------------

    cb_scores = (
        user_profiles[user_id]
        @ item_features.T
    ).toarray().ravel()

    # ----------------------------------------------
    # Normalize scores
    # ----------------------------------------------

    def min_max(scores):

        minimum = np.min(scores)
        maximum = np.max(scores)

        if maximum - minimum < 1e-12:
            return np.zeros_like(scores, dtype=np.float32)

        return (
            (scores - minimum) / (maximum - minimum)
        ).astype(np.float32)

    cf_normalized = min_max(cf_scores)
    cb_normalized = min_max(cb_scores)

    # ----------------------------------------------
    # Hybrid score
    # ----------------------------------------------

    hybrid_scores = (
        alpha * cf_normalized
        + (1.0 - alpha) * cb_normalized
    )

    # Exclude movies already seen in training.
    watched_items = seen_train[user_id].indices

    hybrid_scores[watched_items] = -np.inf

    # ----------------------------------------------
    # Top-K recommendations
    # ----------------------------------------------

    top_indices = np.argsort(
        hybrid_scores
    )[::-1][:top_k]

    recommendations = aligned_movies.iloc[
        top_indices
    ][["movieId", "title", "genres"]].copy()

    recommendations["Hybrid Score"] = hybrid_scores[
        top_indices
    ]

    print("\n========== TOP RECOMMENDED MOVIES ==========")

    for rank, (_, movie) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        print(f"\n{rank}. {movie['title']}")
        print("   Genres:", movie["genres"])
        print(
            "   Hybrid Score:",
            round(movie["Hybrid Score"], 4)
        )

    return recommendations


# ==================================================
# 7. USER INPUT
# ==================================================

if __name__ == "__main__":

    print("\n========== PERSONALIZED MOVIE RECOMMENDER ==========")

    print(
        "Enter an original MovieLens user ID "
        "(not the internal index)."
    )

    try:
        user_id = int(
            input("\nEnter MovieLens User ID: ")
        )

        recommend_movies(
            original_user_id=user_id,
            top_k=10
        )

    except ValueError:
        print("\nPlease enter a valid integer user ID.")
