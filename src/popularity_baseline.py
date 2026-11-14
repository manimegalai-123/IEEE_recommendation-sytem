
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA = BASE_DIR / "dataset" / "processed" / "ml25m_subset.pkl"

RESULTS = BASE_DIR / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

with open(DATA, "rb") as f:
    D = pickle.load(f)

nu = D["n_users"]
ni = D["n_items"]

train = D["train"]
val = D["val"]
test = D["test"]

print("Users:", nu)
print("Items:", ni)

# --------------------------------------------------
# 3. CREATE SEEN-ITEM MATRIX
# --------------------------------------------------

def to_csr(pairs):
    return csr_matrix(
        (
            np.ones(len(pairs), dtype=np.float32),
            (pairs[:, 0], pairs[:, 1])
        ),
        shape=(nu, ni)
    )

seen_train = to_csr(train)
seen_train_val = to_csr(np.vstack([train, val]))

# --------------------------------------------------
# 4. CALCULATE MOVIE POPULARITY
# --------------------------------------------------

# Number of positive training interactions per movie.

popularity = np.bincount(
    train[:, 1],
    minlength=ni
).astype(np.float32)

# --------------------------------------------------
# 5. EVALUATION
# --------------------------------------------------

def evaluate(seen, targets, k=10):

    target_by_user = np.full(nu, -1, dtype=np.int64)
    target_by_user[targets[:, 0]] = targets[:, 1]

    if np.any(target_by_user < 0):
        raise ValueError("Missing target for some users.")

    hits = []
    ndcgs = []

    for user in range(nu):

        scores = popularity.copy()

        # Do not recommend already-seen movies.
        start = seen.indptr[user]
        end = seen.indptr[user + 1]

        seen_items = seen.indices[start:end]
        scores[seen_items] = -1

        target_item = target_by_user[user]
        target_score = scores[target_item]

        # Deterministic tie-breaking by item ID.
        better = (
            (scores > target_score)
            | (
                (scores == target_score)
                & (np.arange(ni) < target_item)
            )
        )

        rank = int(np.sum(better))

        hit = rank < k

        hits.append(float(hit))

        if hit:
            ndcgs.append(1.0 / np.log2(rank + 2))
        else:
            ndcgs.append(0.0)

    return np.mean(hits), np.mean(ndcgs)

# --------------------------------------------------
# 6. VALIDATION
# --------------------------------------------------

val_hr, val_ndcg = evaluate(
    seen_train,
    val,
    k=10
)

print("\nVALIDATION RESULTS")
print("HR@10:", round(val_hr, 4))
print("NDCG@10:", round(val_ndcg, 4))

# --------------------------------------------------
# 7. TEST
# --------------------------------------------------

test_hr, test_ndcg = evaluate(
    seen_train_val,
    test,
    k=10
)

print("\nTEST RESULTS")
print("HR@10:", round(test_hr, 4))
print("NDCG@10:", round(test_ndcg, 4))

# --------------------------------------------------
# 8. SAVE RESULTS
# --------------------------------------------------

metrics = {
    "model": "Popularity Baseline",
    "validation_HR@10": float(val_hr),
    "validation_NDCG@10": float(val_ndcg),
    "test_HR@10": float(test_hr),
    "test_NDCG@10": float(test_ndcg)
}

with open(RESULTS / "popularity_baseline.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nResults saved successfully!")
print("File:", RESULTS / "popularity_baseline.json")
