
import argparse
import pickle
from pathlib import Path

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "dataset" / "processed" / "ml25m_subset.pkl"

NUM_CLIENTS = 5
EMBEDDING_DIM = 32
BATCH_SIZE = 2048
LEARNING_RATE = 0.01
REGULARIZATION = 1e-5

DEVICE = torch.device("cpu")

torch.manual_seed(42)
np.random.seed(42)


# ==========================================
# LOAD CLIENT DATA
# ==========================================

def load_client_data(client_id):

    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    train = np.asarray(data["train"], dtype=np.int64)

    n_users = int(data["n_users"])
    n_items = int(data["n_items"])

    # Assign users to clients.
    # Every user belongs to exactly one client.
    client_interactions = train[
        train[:, 0] % NUM_CLIENTS == client_id
    ]

    if len(client_interactions) == 0:
        raise ValueError(
            f"No training interactions for client {client_id}"
        )

    # Convert global user IDs into local user IDs.
    unique_users = np.unique(client_interactions[:, 0])

    user_mapping = {
        int(user): index
        for index, user in enumerate(unique_users)
    }

    local_users = np.array(
        [
            user_mapping[int(user)]
            for user in client_interactions[:, 0]
        ],
        dtype=np.int64
    )

    positive_items = client_interactions[:, 1].astype(
        np.int64
    )

    # Build positive-item sets for negative sampling.
    user_positive_items = {}

    for user, item in zip(local_users, positive_items):

        user_positive_items.setdefault(
            int(user), set()
        ).add(int(item))

    return (
        n_users,
        n_items,
        len(unique_users),
        local_users,
        positive_items,
        user_positive_items
    )


# ==========================================
# BPR MODEL
# ==========================================

class BPRModel(nn.Module):

    def __init__(self, local_users, n_items):

        super().__init__()

        self.user_embedding = nn.Embedding(
            local_users,
            EMBEDDING_DIM
        )

        self.item_embedding = nn.Embedding(
            n_items,
            EMBEDDING_DIM
        )

        nn.init.normal_(
            self.user_embedding.weight,
            std=0.01
        )

        nn.init.normal_(
            self.item_embedding.weight,
            std=0.01
        )

    def forward(self, users, positive, negative):

        user_vector = self.user_embedding(users)

        positive_vector = self.item_embedding(positive)
        negative_vector = self.item_embedding(negative)

        positive_score = torch.sum(
            user_vector * positive_vector,
            dim=1
        )

        negative_score = torch.sum(
            user_vector * negative_vector,
            dim=1
        )

        return positive_score, negative_score


# ==========================================
# FLOWER CLIENT
# ==========================================

class FederatedBPRClient(fl.client.NumPyClient):

    def __init__(self, client_id):

        self.client_id = client_id

        (
            self.n_users,
            self.n_items,
            self.local_user_count,
            self.users,
            self.positive_items,
            self.user_positive_items
        ) = load_client_data(client_id)

        self.model = BPRModel(
            self.local_user_count,
            self.n_items
        ).to(DEVICE)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=REGULARIZATION
        )

        self.rng = np.random.default_rng(1000 + client_id)

        print(f"\nClient {client_id} initialized")
        print(f"Local users: {self.local_user_count}")
        print(f"Local interactions: {len(self.users)}")

    # --------------------------------------
    # RECEIVE GLOBAL PARAMETERS
    # --------------------------------------

    def get_parameters(self, config):

        return [
            self.model.item_embedding.weight.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        ]

    # --------------------------------------
    # SET GLOBAL PARAMETERS
    # --------------------------------------

    def set_parameters(self, parameters):

        item_weights = torch.tensor(
            parameters[0],
            dtype=torch.float32,
            device=DEVICE
        )

        if tuple(item_weights.shape) != (
            self.n_items,
            EMBEDDING_DIM
        ):
            raise ValueError("Received item embedding shape mismatch")

        with torch.no_grad():

            self.model.item_embedding.weight.copy_(
                item_weights
            )

    # --------------------------------------
    # NEGATIVE SAMPLING
    # --------------------------------------

    def sample_negative_items(self, users):

        negatives = []

        for user in users:

            user = int(user)

            negative = int(
                self.rng.integers(0, self.n_items)
            )

            attempts = 0

            while (
                negative in self.user_positive_items[user]
                and attempts < 100
            ):

                negative = int(
                    self.rng.integers(0, self.n_items)
                )

                attempts += 1

            negatives.append(negative)

        return np.asarray(negatives, dtype=np.int64)

    # --------------------------------------
    # LOCAL TRAINING
    # --------------------------------------

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        local_epochs = int(
            config.get("local_epochs", 1)
        )

        dataset = TensorDataset(
            torch.tensor(self.users, dtype=torch.long),
            torch.tensor(
                self.positive_items,
                dtype=torch.long
            )
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        total_loss = 0.0
        batches = 0

        self.model.train()

        for epoch in range(local_epochs):

            for batch_users, batch_positive in loader:

                batch_users = batch_users.to(DEVICE)
                batch_positive = batch_positive.to(DEVICE)

                negative_items = self.sample_negative_items(
                    batch_users.cpu().numpy()
                )

                batch_negative = torch.tensor(
                    negative_items,
                    dtype=torch.long,
                    device=DEVICE
                )

                self.optimizer.zero_grad()

                positive_score, negative_score = self.model(
                    batch_users,
                    batch_positive,
                    batch_negative
                )

                bpr_loss = -torch.mean(
                    torch.nn.functional.logsigmoid(
                        positive_score - negative_score
                    )
                )

                loss = bpr_loss

                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()
                batches += 1

        average_loss = total_loss / max(batches, 1)

        print(
            f"Client {self.client_id} | "
            f"Local loss: {average_loss:.4f}"
        )

        return (
            self.get_parameters({}),
            len(self.users),
            {
                "loss": float(average_loss)
            }
        )

    # --------------------------------------
    # EVALUATION
    # --------------------------------------

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        return 0.0, len(self.users), {}


# ==========================================
# START CLIENT
# ==========================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--client_id",
        type=int,
        required=True
    )

    parser.add_argument(
        "--server",
        type=str,
        default="127.0.0.1:8080"
    )

    args = parser.parse_args()

    if not 0 <= args.client_id < NUM_CLIENTS:
        raise ValueError("Client ID must be between 0 and 4")

    client = FederatedBPRClient(args.client_id)

    print(
        f"\nConnecting Client {args.client_id} "
        f"to server {args.server}"
    )

    fl.client.start_client(
        server_address=args.server,
        client=client.to_client()
    )
