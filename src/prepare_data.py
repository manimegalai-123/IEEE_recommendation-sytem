
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

# Find project root automatically
BASE_DIR = Path(__file__).resolve().parent.parent

# Input dataset
RAW = BASE_DIR / "dataset" / "raw" / "ml-25m" / "ratings.csv"

# Output directory
OUT = BASE_DIR / "dataset" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

N_USERS = 10000
MIN_USER = 20
MIN_ITEM = 5
POS_THRESH = 4.0
SEED = 42

# --------------------------------------------------
# 3. LOAD RAW DATA
# --------------------------------------------------

if not RAW.exists():
    raise FileNotFoundError(
        f"Ratings dataset not found at: {RAW}\n"
        "Check that ratings.csv is inside dataset/raw/ml-25m/"
    )

df = pd.read_csv(
    RAW,
    dtype={
        "userId": "int32",
        "movieId": "int32",
        "rating": "float32",
        "timestamp": "int64"
    }
)

required_columns = {"userId", "movieId", "rating", "timestamp"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Missing columns: {required_columns - set(df.columns)}"
    )

print("Raw ratings:", len(df))

# --------------------------------------------------
# 4. CONVERT RATINGS INTO IMPLICIT FEEDBACK
# --------------------------------------------------

# Ratings >= 4.0 are considered positive interactions.
# We retain only userId, movieId and timestamp.

df = df.loc[
    df["rating"] >= POS_THRESH,
    ["userId", "movieId", "timestamp"]
].copy()

print("Positive interactions:", len(df))

# Remove duplicate user-movie interactions, if any.
df = df.drop_duplicates(subset=["userId", "movieId"])

# --------------------------------------------------
# 5. SAMPLE USERS
# --------------------------------------------------

# Keep users with at least MIN_USER positive interactions.

user_counts = df["userId"].value_counts()

eligible_users = user_counts[
    user_counts >= MIN_USER
].index.to_numpy()

if len(eligible_users) == 0:
    raise ValueError(
        "No users have enough positive interactions. "
        "Reduce MIN_USER or check the dataset."
    )

rng = np.random.default_rng(SEED)

selected_users = rng.choice(
    eligible_users,
    size=min(N_USERS, len(eligible_users)),
    replace=False
)

df = df[df["userId"].isin(selected_users)].copy()

print("Interactions after user sampling:", len(df))
print("Sampled users:", df["userId"].nunique())

# --------------------------------------------------
# 6. ITERATIVE K-CORE FILTERING
# --------------------------------------------------

# Every user must have at least MIN_USER interactions.
# Every movie must have at least MIN_ITEM interactions.
# Repeat until no more rows are removed.

while True:

    previous_size = len(df)

    # Filter movies with insufficient interactions
    movie_counts = df.groupby("movieId")["userId"].transform("size")
    df = df[movie_counts >= MIN_ITEM]

    # Filter users with insufficient interactions
    user_counts = df.groupby("userId")["movieId"].transform("size")
    df = df[user_counts >= MIN_USER]

    if len(df) == previous_size:
        break

if df.empty:
    raise ValueError(
        "No interactions remain after k-core filtering. "
        "Try reducing MIN_USER or MIN_ITEM."
    )

print("Interactions after k-core filtering:", len(df))

# --------------------------------------------------
# 7. RE-INDEX USER AND MOVIE IDS
# --------------------------------------------------

# Convert original IDs into continuous integer IDs
# starting from zero.

user_codes, user_ids = pd.factorize(df["userId"], sort=True)
item_codes, movie_ids = pd.factorize(df["movieId"], sort=True)

df["u"] = user_codes.astype("int32")
df["i"] = item_codes.astype("int32")

n_users = len(user_ids)
n_items = len(movie_ids)

# Save mappings to recover original IDs later.
user_map = {
    index: int(original_id)
    for index, original_id in enumerate(user_ids)
}

item_map = {
    index: int(original_id)
    for index, original_id in enumerate(movie_ids)
}

print("Final users:", n_users)
print("Final movies:", n_items)

# --------------------------------------------------
# 8. TIME-BASED TRAIN / VALIDATION / TEST SPLIT
# --------------------------------------------------

# Sort each user's interactions chronologically.
df = df.sort_values(
    ["u", "timestamp"],
    kind="mergesort"
).reset_index(drop=True)

# Last interaction -> test
# Second-last interaction -> validation
# Remaining interactions -> training

rank_from_end = df.groupby("u").cumcount(ascending=False)

test = df[rank_from_end == 0]
val = df[rank_from_end == 1]
train = df[rank_from_end >= 2]

# Convert arrays to compact integer types.
train_data = train[["u", "i"]].to_numpy(dtype=np.int32)
val_data = val[["u", "i"]].to_numpy(dtype=np.int32)
test_data = test[["u", "i"]].to_numpy(dtype=np.int32)

# --------------------------------------------------
# 9. SAVE PROCESSED DATASET
# --------------------------------------------------

processed_data = {
    "n_users": n_users,
    "n_items": n_items,
    "train": train_data,
    "val": val_data,
    "test": test_data,
    "user_map": user_map,
    "item_map": item_map
}

output_file = OUT / "ml25m_subset.pkl"

with open(output_file, "wb") as f:
    pickle.dump(processed_data, f, protocol=pickle.HIGHEST_PROTOCOL)

# --------------------------------------------------
# 10. DISPLAY FINAL RESULTS
# --------------------------------------------------

print("\n========== DATA PREPARATION COMPLETE ==========")

print(f"Users       : {n_users}")
print(f"Movies      : {n_items}")
print(f"Train       : {len(train_data)}")
print(f"Validation  : {len(val_data)}")
print(f"Test        : {len(test_data)}")

print(f"\nSaved file: {output_file}")

print("\nDataset preparation completed successfully!")
