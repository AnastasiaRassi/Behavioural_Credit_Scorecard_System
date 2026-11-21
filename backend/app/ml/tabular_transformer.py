"""Does a transformer on the flat engineered features beat the booster?

`sequence_models.py` already tried a transformer encoder over the six monthly timesteps. This
is the other kind: an FT-Transformer (Gorishniy et al., 2021), which treats each engineered
feature as a token and lets self-attention mix them. It reads exactly what the deployed
booster reads, so the comparison is about the model family and nothing else.

Each numeric feature is quantile-scaled on train rows only, then turned into a token by its
own linear map, with a learned [CLS] token whose output feeds the classifier. The comparison
set is a tuned-for-validation search of four configurations, three seeds on the winner, the
deployed booster, and logistic regression on the same scaled features.

The blend weight is chosen on validation and applied once to test, as in `sequence_models.py`.
Nothing here is a candidate for deployment; `train.py` builds the served model.

Run `python -m app.ml.tabular_transformer` to regenerate `tabular_transformer.json`.
"""
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from torch import nn

from app.ml.sequence_models import (BATCH, BLEND_WEIGHTS, EPOCHS, PATIENCE, SEED, Data,
                                    artifacts_dir, booster_scores, ranked)
from app.ml.train import SEEDS

CONFIGURATIONS = [
    {"token_size": 32, "layers": 2, "dropout": 0.1, "learning_rate": 1e-3},
    {"token_size": 64, "layers": 2, "dropout": 0.2, "learning_rate": 5e-4},
    {"token_size": 32, "layers": 3, "dropout": 0.2, "learning_rate": 3e-4},
    {"token_size": 16, "layers": 1, "dropout": 0.1, "learning_rate": 3e-3},
]


class FeatureTransformer(nn.Module):
    """FT-Transformer: one token per feature, a [CLS] token, a pre-norm encoder."""

    def __init__(self, features: int, token_size: int, layers: int, dropout: float):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(features, token_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(features, token_size))
        self.cls = nn.Parameter(torch.randn(1, 1, token_size) * 0.1)
        layer = nn.TransformerEncoderLayer(
            token_size, nhead=4, dim_feedforward=token_size * 2, dropout=dropout,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(token_size), nn.ReLU(),
                                  nn.Linear(token_size, 1))

    def forward(self, values):
        tokens = values.unsqueeze(2) * self.weight + self.bias
        tokens = torch.cat([self.cls.expand(len(values), -1, -1), tokens], dim=1)
        return self.head(self.encoder(tokens)[:, 0, :]).squeeze(1)


def scaled_features(data: Data) -> dict:
    """Quantile scaling fitted on train only, so nothing leaks into validation or test."""
    scaler = QuantileTransformer(n_quantiles=500, output_distribution="normal",
                                 random_state=SEED)
    scaler.fit(data.flat.loc[data.index["train"]])
    # Engineered ratios can be NaN (the booster handles that natively); 0 is the median
    # after normal scaling, so a missing value reads as "typical".
    return {name: np.nan_to_num(scaler.transform(data.flat.loc[index])).astype(np.float32)
            for name, index in data.index.items()}


def train_network(features: dict, data: Data, token_size: int, layers: int, dropout: float,
                  learning_rate: float, seed: int):
    """Fit on train, early stopping on validation AUC. Returns the best model and score."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FeatureTransformer(features["train"].shape[1], token_size, layers, dropout)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_function = nn.BCEWithLogitsLoss()

    inputs = torch.from_numpy(features["train"])
    target = torch.from_numpy(data.y["train"])
    validation = torch.from_numpy(features["validation"])

    best_auc, best_epoch, best_state, waited = 0.0, 0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(target))
        for start in range(0, len(order), BATCH):
            picked = order[start:start + BATCH]
            optimiser.zero_grad()
            loss_function(model(inputs[picked]), target[picked]).backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(validation)).numpy()
        auc = roc_auc_score(data.y["validation"], scores)
        if auc > best_auc:
            best_auc, best_epoch, waited = auc, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, float(best_auc), best_epoch


def network_scores(model, features: dict, name: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(features[name]))).numpy()


def main() -> dict:
    torch.set_num_threads(4)
    data = Data()
    features = scaled_features(data)
    booster = booster_scores(data)
    booster_validation = float(roc_auc_score(data.y["validation"], booster["validation"]))
    booster_test = float(roc_auc_score(data.y["test"], booster["test"]))
    print(f"booster validation {booster_validation:.4f}  test {booster_test:.4f}", flush=True)

    logistic = LogisticRegression(max_iter=2000).fit(features["train"], data.y["train"])
    logistic_validation = float(roc_auc_score(
        data.y["validation"], logistic.predict_proba(features["validation"])[:, 1]))
    logistic_test = float(roc_auc_score(
        data.y["test"], logistic.predict_proba(features["test"])[:, 1]))
    print(f"logistic validation {logistic_validation:.4f}  test {logistic_test:.4f}", flush=True)

    search = []
    for configuration in CONFIGURATIONS:
        _, auc, epoch = train_network(features, data, seed=SEED, **configuration)
        search.append({**configuration, "validation_auc": auc, "epochs": epoch})
        print(f"  {configuration} -> {auc:.4f}", flush=True)
    best = max(search, key=lambda row: row["validation_auc"])
    settings = {k: best[k] for k in ("token_size", "layers", "dropout", "learning_rate")}

    validation_scores, test_scores, seeds_used = [], [], []
    for seed in SEEDS:
        model, auc, epoch = train_network(features, data, seed=seed, **settings)
        validation_scores.append(network_scores(model, features, "validation"))
        test_scores.append(network_scores(model, features, "test"))
        seeds_used.append({"seed": seed, "validation_auc": auc, "epochs": epoch,
                           "test_auc": float(roc_auc_score(data.y["test"], test_scores[-1]))})
        print(f"  seed {seed}: validation {auc:.4f}  test {seeds_used[-1]['test_auc']:.4f}",
              flush=True)

    network = {"validation": np.mean(validation_scores, axis=0),
               "test": np.mean(test_scores, axis=0)}
    network_validation = float(roc_auc_score(data.y["validation"], network["validation"]))
    network_test = float(roc_auc_score(data.y["test"], network["test"]))
    correlation = float(np.corrcoef(ranked(booster["validation"]),
                                    ranked(network["validation"]))[0, 1])

    sweep = {}
    for weight in BLEND_WEIGHTS:
        blended = (1 - weight) * ranked(booster["validation"]) + weight * ranked(
            network["validation"])
        sweep[float(weight)] = float(roc_auc_score(data.y["validation"], blended))
    best_weight = max(sweep, key=sweep.get)
    blended_test = (1 - best_weight) * ranked(booster["test"]) + best_weight * ranked(
        network["test"])
    blend_test_auc = float(roc_auc_score(data.y["test"], blended_test))

    results = {
        "split_seed": SEED,
        "model_seeds": list(SEEDS),
        "epochs_max": EPOCHS,
        "patience": PATIENCE,
        "batch": BATCH,
        "features": int(features["train"].shape[1]),
        "booster": {"validation_auc": booster_validation, "test_auc": booster_test},
        "logistic": {"validation_auc": logistic_validation, "test_auc": logistic_test},
        "search": search,
        "winner": best,
        "winner_seeds": seeds_used,
        "network": {"validation_auc": network_validation, "test_auc": network_test},
        "rank_correlation_with_booster": correlation,
        "blend_sweep_validation": sweep,
        "blend": {
            "weight": float(best_weight),
            "validation_auc": sweep[best_weight],
            "validation_gain": sweep[best_weight] - booster_validation,
            "test_auc": blend_test_auc,
            "test_gain": blend_test_auc - booster_test,
        },
    }

    out = artifacts_dir() / "tabular_transformer.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1)
    print(f"\nnetwork validation {network_validation:.4f}  test {network_test:.4f}", flush=True)
    print(f"blend weight {best_weight}: validation {sweep[best_weight]:.4f} "
          f"({sweep[best_weight] - booster_validation:+.4f}), "
          f"test {blend_test_auc:.4f} ({blend_test_auc - booster_test:+.4f})", flush=True)
    print(f"wrote {out}", flush=True)
    return results


if __name__ == "__main__":
    main()
