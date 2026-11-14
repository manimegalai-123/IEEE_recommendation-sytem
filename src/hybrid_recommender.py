
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.sparse import csr_matrix, load_npz, vstack
from sklearn.preprocessing import normalize


# ==================================================
# 1. PATH CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "dataset" / "processed" / "ml25m_subset.pkl"

TFIDF_FILE = BASE_DIR / "dataset" / "processed" / "movie_tfidf.npz"

MOVIE_INDEX_FILE = (
    BASE_DIR / "dataset" / "processed" / "content_movie_index.csv"
)

MODEL_FILE = BASE_DIR / "results" / "baseline_mf.pt"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "hybrid_recommender.json"

device = torch.device("cpu")


# ==================================================
# 2. LOAD DATA
# ==================================================

print("\n========== LOADING DATA ==========")

with open(DATA_FILE, "rb") as f:
    D = pickle.load(f)

nu = D["n_users"]
ni = D["n_items"]

train = D["train"]
val = D["val"]
test = D["test"]

print("Users:", nu)
print("Items:", ni)
print("Training interactions:", len(train))
print("Validation interactions:", len(val))
print("Test interactions:", len(test))


# ==================================================
# 3. LOAD TRAINED MATRIX FACTORIZATION MODEL
# ==================================================

print("\n========== LOADING BPR MODEL ==========")

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

if checkpoint["n_users"] != nu or checkpoint["n_items"] != ni:
    raise ValueError(
        "Checkpoint dimensions do not match the processed dataset."
    )

embedding_dim = checkpoint["embedding_dim"]

user_embeddings = (
    checkpoint["model_state_dict"]["U.weight"]
    .to(device)
)

item_embeddings = (
    checkpoint["model_state_dict"]["I.weight"]
    .to(device)
)

print("BPR model loaded successfully!")
print("Embedding dimension:", embedding_dim)


# ==================================================
# 4. LOAD TF-IDF FEATURES
# ==================================================

print("\n========== LOADING CONTENT FEATURES ==========")

tfidf_all = load_npz(TFIDF_FILE).tocsr()

movie_index = pd.read_csv(MOVIE_INDEX_FILE)

movie_index["movieId"] = movie_index["movieId"].astype("int64")

print("Complete TF-IDF shape:", tfidf_all.shape)
print("Movie index records:", len(movie_index))


# ==================================================
# 5. MAP MODEL ITEM IDs TO TF-IDF ROWS
# ==================================================

print("\n========== MAPPING MOVIE IDS ==========")

# item_map:
# Reindexed item ID -> original MovieLens movieId

item_map = D["item_map"]

if isinstance(item_map, dict):
    original_movie_ids = [
        item_map.get(i, item_map.get(str(i)))
        for i in range(ni)
    ]
else:
    original_movie_ids = list(item_map)

if len(original_movie_ids) != ni:
    raise ValueError("item_map size does not match model item count.")

if any(movie_id is None for movie_id in original_movie_ids):
    raise ValueError("Some original MovieLens movie IDs are missing.")

original_movie_ids = [
    int(movie_id) for movie_id in original_movie_ids
]

# Original MovieLens movieId -> TF-IDF row
movie_to_row = {
    int(movie_id): row
    for row, movie_id in enumerate(movie_index["movieId"])
}

# Build feature matrix in the same item order as BPR.
feature_rows = []
missing_movies = 0
n_features = tfidf_all.shape[1]

for movie_id in original_movie_ids:

    row = movie_to_row.get(movie_id)

    if row is None:
        feature_rows.append(csr_matrix((1, n_features)))
        missing_movies += 1
    else:
        feature_rows.append(tfidf_all[row])

item_features = vstack(feature_rows, format="csr")

print("Model items:", ni)
print("Mapped items:", ni - missing_movies)
print("Missing metadata:", missing_movies)
print("Aligned feature matrix:", item_features.shape)

if missing_movies == ni:
    raise ValueError("No model movies could be mapped to TF-IDF features.")


# ==================================================
# 6. CREATE USER-ITEM MATRICES
# ==================================================

def to_csr(pairs):

    return csr_matrix(
        (
            np.ones(len(pairs), dtype=np.float32),
            (pairs[:, 0], pairs[:, 1])
        ),
        shape=(nu, ni)
    )


seen_train = to_csr(train)

seen_train_val = to_csr(
    np.vstack([train, val])
)


# ==================================================
# 7. CREATE USER CONTENT PROFILES
# ==================================================

print("\n========== BUILDING USER CONTENT PROFILES ==========")

# Each user profile is the average content representation
# of movies the user liked in the training dataset.

user_profiles = seen_train @ item_features

# L2 normalization makes cosine similarity meaningful.
user_profiles = normalize(
    user_profiles,
    norm="l2",
    axis=1
).tocsr()

print("User profiles created:", user_profiles.shape)


# ==================================================
# 8. SCORE NORMALIZATION
# ==================================================

def normalize_scores(scores):

    """
    Min-max normalize scores separately for each user.
    Masked items remain unavailable.
    """

    valid = np.isfinite(scores)

    minimum = np.min(
        np.where(valid, scores, np.inf),
        axis=1,
        keepdims=True
    )

    maximum = np.max(
        np.where(valid, scores, -np.inf),
        axis=1,
        keepdims=True
    )

    denominator = maximum - minimum

    normalized = np.zeros_like(scores, dtype=np.float32)

    np.divide(
        scores - minimum,
        denominator,
        out=normalized,
        where=valid & (denominator > 1e-12)
    )

    normalized[~valid] = -np.inf

    return normalized


# ==================================================
# 9. HYBRID EVALUATION
# ==================================================

def evaluate_hybrid(
    seen,
    targets,
    alpha,
    k=10,
    batch_size=100
):

    """
    Evaluate the hybrid recommendation model.

    alpha:
        Weight given to collaborative filtering.

    1-alpha:
        Weight given to content-based filtering.
    """

    target_by_user = np.full(
        nu,
        -1,
        dtype=np.int64
    )

    target_by_user[targets[:, 0]] = targets[:, 1]

    if np.any(target_by_user < 0):
        raise ValueError("Some users have no evaluation target.")

    hits = []
    ndcgs = []

    user_embeddings_np = user_embeddings.detach().numpy()
    item_embeddings_np = item_embeddings.detach().numpy()

    for start in range(0, nu, batch_size):

        users = np.arange(
            start,
            min(start + batch_size, nu)
        )

        # ------------------------------------------
        # Collaborative filtering scores
        # ------------------------------------------

        cf_scores = (
            user_embeddings_np[users]
            @ item_embeddings_np.T
        )

        # ------------------------------------------
        # Content-based cosine similarity scores
        # ------------------------------------------

        cb_scores = (
            user_profiles[users]
            @ item_features.T
        ).toarray().astype(np.float32)

        # ------------------------------------------
        # Mask previously watched/liked movies
        # ------------------------------------------

        seen_batch = seen[users].tocoo()

        cf_scores[
            seen_batch.row,
            seen_batch.col
        ] = -np.inf

        cb_scores[
            seen_batch.row,
            seen_batch.col
        ] = -np.inf

        # ------------------------------------------
        # Normalize both scoring methods
        # ------------------------------------------

        cf_normalized = normalize_scores(cf_scores)

        cb_normalized = normalize_scores(cb_scores)

        # ------------------------------------------
        # Combine scores
        # ------------------------------------------

        hybrid_scores = (
            alpha * cf_normalized
            + (1.0 - alpha) * cb_normalized
        )

        # ------------------------------------------
        # Calculate target ranks
        # ------------------------------------------

        target_items = target_by_user[users]

        target_scores = hybrid_scores[
            np.arange(len(users)),
            target_items
        ]

        ranks = (
            hybrid_scores > target_scores[:, None]
        ).sum(axis=1)

        hit = (ranks < k).astype(np.float32)

        ndcg = hit / np.log2(ranks + 2)

        hits.extend(hit.tolist())
        ndcgs.extend(ndcg.tolist())

    return (
        float(np.mean(hits)),
        float(np.mean(ndcgs))
    )


# ==================================================
# 10. TUNE ALPHA USING VALIDATION DATA
# ==================================================

print("\n========== ALPHA TUNING ==========")

best_alpha = None
best_val_hr = -1
best_val_ndcg = -1

alpha_results = []

for alpha in np.arange(0.0, 1.01, 0.05):

    alpha = round(float(alpha), 2)

    hr, ndcg = evaluate_hybrid(
        seen=seen_train,
        targets=val,
        alpha=alpha,
        k=10
    )

    alpha_results.append({
        "alpha": alpha,
        "HR@10": hr,
        "NDCG@10": ndcg
    })

    print(
        f"Alpha: {alpha:.2f} | "
        f"Val HR@10: {hr:.4f} | "
        f"Val NDCG@10: {ndcg:.4f}"
    )

    # Select using validation NDCG.
    if ndcg > best_val_ndcg:

        best_val_ndcg = ndcg
        best_val_hr = hr
        best_alpha = alpha


print("\nBest alpha:", best_alpha)
print("Best validation HR@10:", best_val_hr)
print("Best validation NDCG@10:", best_val_ndcg)


# ==================================================
# 11. FINAL TEST EVALUATION
# ==================================================

print("\n========== FINAL TEST EVALUATION ==========")

test_hr, test_ndcg = evaluate_hybrid(
    seen=seen_train_val,
    targets=test,
    alpha=best_alpha,
    k=10
)

print("Selected alpha:", best_alpha)
print(f"Test HR@10: {test_hr:.4f}")
print(f"Test NDCG@10: {test_ndcg:.4f}")


# ==================================================
# 12. SAVE RESULTS
# ==================================================

results = {
    "model": "Hybrid BPR Matrix Factorization + TF-IDF",
    "embedding_dim": embedding_dim,
    "alpha": best_alpha,
    "content_weight": round(1.0 - best_alpha, 2),
    "validation": {
        "HR@10": best_val_hr,
        "NDCG@10": best_val_ndcg
    },
    "test": {
        "HR@10": test_hr,
        "NDCG@10": test_ndcg
    },
    "alpha_experiments": alpha_results
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=4)

print("\nHybrid results saved successfully!")
print("File:", OUTPUT_FILE)

print("\n========== HYBRID MODEL COMPLETE ==========")
