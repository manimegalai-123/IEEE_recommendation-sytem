
import pickle
from pathlib import Path

import flwr as fl
import numpy as np

from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays
)


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "dataset" / "processed" / "ml25m_subset.pkl"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "federated_item_embeddings.npy"

NUM_CLIENTS = 5
NUM_ROUNDS = 5
EMBEDDING_DIM = 32


# ==========================================
# LOAD DATASET INFORMATION
# ==========================================

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

N_ITEMS = int(data["n_items"])


# ==========================================
# INITIALIZE GLOBAL ITEM EMBEDDINGS
# ==========================================

rng = np.random.default_rng(42)

initial_item_embeddings = rng.normal(
    loc=0.0,
    scale=0.01,
    size=(N_ITEMS, EMBEDDING_DIM)
).astype(np.float32)

initial_parameters = ndarrays_to_parameters(
    [initial_item_embeddings]
)


# ==========================================
# CUSTOM FEDAVG STRATEGY
# ==========================================

class SavingFedAvg(fl.server.strategy.FedAvg):

    def aggregate_fit(self, server_round, results, failures):

        aggregated_parameters, aggregated_metrics = (
            super().aggregate_fit(
                server_round,
                results,
                failures
            )
        )

        if aggregated_parameters is not None:

            arrays = parameters_to_ndarrays(
                aggregated_parameters
            )

            item_embeddings = arrays[0]

            np.save(
                OUTPUT_PATH,
                item_embeddings
            )

            print("\n========== FEDERATED ROUND ==========")

            print("Round:", server_round)

            print("Participating clients:", len(results))

            print(
                "Global item embedding shape:",
                item_embeddings.shape
            )

            print(
                "Saved model:",
                OUTPUT_PATH
            )

        return aggregated_parameters, aggregated_metrics


# ==========================================
# FEDAVG STRATEGY
# ==========================================

strategy = SavingFedAvg(

    fraction_fit=1.0,

    fraction_evaluate=0.0,

    min_fit_clients=NUM_CLIENTS,

    min_evaluate_clients=NUM_CLIENTS,

    min_available_clients=NUM_CLIENTS,

    initial_parameters=initial_parameters,

    on_fit_config_fn=lambda server_round: {
        "local_epochs": 1
    }
)


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    print("\n====================================")
    print(" FEDERATED BPR RECOMMENDATION SYSTEM")
    print("====================================")

    print("Users:", data["n_users"])
    print("Items:", N_ITEMS)
    print("Clients:", NUM_CLIENTS)
    print("Federated rounds:", NUM_ROUNDS)

    print("\nStarting Flower server...")

    fl.server.start_server(

        server_address="0.0.0.0:8080",

        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),

        strategy=strategy
    )
