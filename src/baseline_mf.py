
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix

# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA = BASE_DIR / "dataset" / "processed" / "ml25m_subset.pkl"

RES = BASE_DIR / "results"
RES.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(
        f"Processed dataset not found: {DATA}\n"
        "Run prepare_data.py first."
    )

# --------------------------------------------------
# 2. REPRODUCIBILITY
# --------------------------------------------------

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# CPU training
device = torch.device("cpu")

# --------------------------------------------------
# 3. LOAD PROCESSED DATA
# --------------------------------------------------

with open(DATA, "rb") as f:
    D = pickle.load(f)

nu = D["n_users"]
ni = D["n_items"]

tr = D["train"]
va = D["val"]
te = D["test"]

print("\n========== DATASET INFORMATION ==========")
print(f"Users: {nu}")
print(f"Items: {ni}")
print(f"Train interactions: {len(tr)}")
print(f"Validation interactions: {len(va)}")
print(f"Test interactions: {len(te)}")

# Verify leave-one-out structure
assert len(va) == nu, "Expected one validation interaction per user."
assert len(te) == nu, "Expected one test interaction per user."

# --------------------------------------------------
# 4. CREATE USER-ITEM SPARSE MATRIX
# --------------------------------------------------

def to_csr(pairs):
    return csr_matrix(
        (
            np.ones(len(pairs), dtype=np.float32),
            (pairs[:, 0], pairs[:, 1])
        ),
        shape=(nu, ni)
    )

seen_train = to_csr(tr)

seen_train_val = to_csr(
    np.vstack([tr, va])
)

# --------------------------------------------------
# 5. MATRIX FACTORIZATION MODEL
# --------------------------------------------------

class MF(nn.Module):

    def __init__(self, n_users, n_items, embedding_dim=32):

        super().__init__()

        self.U = nn.Embedding(n_users, embedding_dim)
        self.I = nn.Embedding(n_items, embedding_dim)

        nn.init.normal_(self.U.weight, std=0.1)
        nn.init.normal_(self.I.weight, std=0.1)

    def forward(self, users, items):

        user_vectors = self.U(users)
        item_vectors = self.I(items)

        scores = (user_vectors * item_vectors).sum(dim=1)

        return scores


# --------------------------------------------------
# 6. NEGATIVE SAMPLING
# --------------------------------------------------


def sample_negative(users):

    """
    Sample movies that users have NOT interacted
    with in the training dataset.
    """

    users_np = users.cpu().numpy().copy()

    negatives = np.random.randint(
        0, ni, size=len(users_np)
    )

    # Check each user's sampled movies against
    # their training interactions.
    for idx, user_id in enumerate(users_np):

        while seen_train[int(user_id), int(negatives[idx])] > 0:

            negatives[idx] = np.random.randint(0, ni)

    return torch.tensor(
        negatives,
        dtype=torch.long,
        device=device
    )



# --------------------------------------------------
# 7. EVALUATION: HR@10 AND NDCG@10
# --------------------------------------------------

@torch.no_grad()
def evaluate(model, seen, target, k=10, batch_size=1000):

    """
    Full-ranking leave-one-out evaluation.

    HR@K:
        Measures whether the held-out movie appears
        in the top K recommendations.

    NDCG@K:
        Gives higher scores when the held-out movie
        appears closer to the top.
    """

    model.eval()

    # Map each user's held-out movie directly by user ID.
    target_by_user = np.full(nu, -1, dtype=np.int64)

    target_by_user[target[:, 0]] = target[:, 1]

    if np.any(target_by_user < 0):
        raise ValueError(
            "Some users do not have a target interaction."
        )

    hits = []
    ndcgs = []

    for start in range(0, nu, batch_size):

        users = np.arange(
            start,
            min(start + batch_size, nu)
        )

        user_tensor = torch.tensor(
            users,
            dtype=torch.long,
            device=device
        )

        # Calculate scores for every item.
        scores = (
            model.U(user_tensor) @ model.I.weight.T
        )

        # Mask items already interacted with.
        seen_batch = seen[users].tocoo()

        scores[
            seen_batch.row,
            seen_batch.col
        ] = -1e9

        # Find target item scores.
        target_items = torch.tensor(
            target_by_user[users],
            dtype=torch.long,
            device=device
        )

        target_scores = scores[
            torch.arange(len(users)),
            target_items
        ]

        # Rank = number of items with higher scores.
        ranks = (
            scores > target_scores.unsqueeze(1)
        ).sum(dim=1).float()

        hit = (ranks < k).float()

        ndcg = hit / torch.log2(ranks + 2)

        hits.append(hit)
        ndcgs.append(ndcg)

    hr = torch.cat(hits).mean().item()
    ndcg = torch.cat(ndcgs).mean().item()

    return hr, ndcg


# --------------------------------------------------
# 8. MODEL TRAINING CONFIGURATION
# --------------------------------------------------

EMBEDDING_DIM = 32
EPOCHS = 20
BATCH_SIZE = 4096
LEARNING_RATE = 0.01
REG = 1e-5

model = MF(
    nu,
    ni,
    embedding_dim=EMBEDDING_DIM
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

train_users = torch.tensor(
    tr[:, 0],
    dtype=torch.long,
    device=device
)

train_items = torch.tensor(
    tr[:, 1],
    dtype=torch.long,
    device=device
)

# --------------------------------------------------
# 9. TRAIN USING BPR LOSS
# --------------------------------------------------

print("\n========== TRAINING STARTED ==========")

for epoch in range(1, EPOCHS + 1):

    model.train()

    permutation = torch.randperm(len(train_users))

    total_loss = 0.0

    for start in range(0, len(permutation), BATCH_SIZE):

        batch_indices = permutation[
            start:start + BATCH_SIZE
        ]

        users = train_users[batch_indices]

        positive_items = train_items[batch_indices]

        # Sample unobserved movies.
        negative_items = sample_negative(users)

        # Embeddings
        user_vectors = model.U(users)

        positive_vectors = model.I(positive_items)

        negative_vectors = model.I(negative_items)

        # Positive and negative prediction scores.
        positive_scores = (
            user_vectors * positive_vectors
        ).sum(dim=1)

        negative_scores = (
            user_vectors * negative_vectors
        ).sum(dim=1)

        # Bayesian Personalized Ranking loss.
        bpr_loss = -F.logsigmoid(
            positive_scores - negative_scores
        ).mean()

        # L2 regularization.
        regularization = REG * (
            user_vectors.pow(2).sum()
            + positive_vectors.pow(2).sum()
            + negative_vectors.pow(2).sum()
        ) / len(users)

        loss = bpr_loss + regularization

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * len(users)

    average_loss = total_loss / len(train_users)

    # Evaluate every two epochs.
    if epoch % 2 == 0:

        hr, ndcg = evaluate(
            model,
            seen_train,
            va,
            k=10
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Val HR@10: {hr:.4f} | "
            f"Val NDCG@10: {ndcg:.4f}"
        )

# --------------------------------------------------
# 10. FINAL TEST EVALUATION
# --------------------------------------------------

print("\n========== FINAL TEST EVALUATION ==========")

test_hr, test_ndcg = evaluate(
    model,
    seen_train_val,
    te,
    k=10
)

print(f"Test HR@10: {test_hr:.4f}")
print(f"Test NDCG@10: {test_ndcg:.4f}")

# --------------------------------------------------
# 11. SAVE MODEL AND RESULTS
# --------------------------------------------------

metrics = {
    "model": "Matrix Factorization with BPR",
    "embedding_dim": EMBEDDING_DIM,
    "epochs": EPOCHS,
    "HR@10": test_hr,
    "NDCG@10": test_ndcg
}

with open(RES / "baseline_mf.json", "w") as f:
    json.dump(metrics, f, indent=4)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "n_users": nu,
        "n_items": ni,
        "embedding_dim": EMBEDDING_DIM,
        "user_map": D.get("user_map"),
        "item_map": D.get("item_map")
    },
    RES / "baseline_mf.pt"
)

print("\nModel saved successfully!")
print("Results saved to:", RES)
print("\n========== TRAINING COMPLETE ==========")
