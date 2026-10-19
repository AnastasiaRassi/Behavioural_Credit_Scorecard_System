"""Does reading the six months as a sequence beat summarising them?

Every other model in the project sees flat per-customer statistics: `pay_max`, `util_mean`,
`repay_ratio_min`. This one reads the monthly matrix in order — six timesteps by three
channels, with the repayment status embedded rather than scaled, because -2 ("no balance"),
-1 ("paid in full") and 0 ("revolving") are categories and -2 is healthier than 0.

Three questions:

1. **Architecture.** Five of them, four configurations each, on validation. The booster had
   seventy tuning configurations behind it; comparing it against one hand-set network would
   not be a fair test.
2. **Blending.** The interesting number is not the network alone, which has far less feature
   engineering behind it, but whether blending it with the deployed booster beats the
   booster. `experiments.py` stacked tree models with each other, all seeing the same flat
   features; a sequence model is a different hypothesis class, which is where ensembles
   usually pay.
3. **Honesty.** The blend weight is chosen on validation and then applied once to test. An
   earlier version of this experiment reported +0.0008 from a weight picked on the same
   split it was scored on; on test that reversed to -0.0014. Selection bias is the finding,
   and it is reported rather than buried.

Nothing here is a candidate for deployment. `train.py` builds the served model.

Run `python -m app.ml.sequence_models` to regenerate `sequence_models.json`.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

from app.ml.feature_engineering import PROTECTED, TARGET, engineer, load_taiwan, project_root
from app.ml.train import HYPERPARAMETERS, SEEDS

SEED = 42
MONTHS = [6, 5, 4, 3, 2, 1]  # oldest first
PAY_STATES = range(-2, 9)
EPOCHS, PATIENCE, BATCH = 40, 6, 512
BLEND_WEIGHTS = np.round(np.arange(0.0, 0.70, 0.05), 2)

ARCHITECTURES = ["gru", "lstm", "bilstm", "cnn", "attention"]
CONFIGURATIONS = [
    {"hidden": 48, "layers": 1, "dropout": 0.2, "learning_rate": 1e-3},
    {"hidden": 96, "layers": 1, "dropout": 0.3, "learning_rate": 1e-3},
    {"hidden": 64, "layers": 2, "dropout": 0.2, "learning_rate": 3e-4},
    {"hidden": 32, "layers": 1, "dropout": 0.1, "learning_rate": 3e-3},
]


def artifacts_dir() -> Path:
    path = project_root() / "backend" / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


class SequenceModel(nn.Module):
    """One body over the embedded months, chosen by `kind`, then a static-aware head."""

    def __init__(self, kind: str, hidden: int, layers: int, dropout: float,
                 embedding: int = 8):
        super().__init__()
        self.kind = kind
        self.embedding = nn.Embedding(len(PAY_STATES), embedding)
        width = embedding + 2
        if kind in ("gru", "lstm", "bilstm"):
            recurrent = nn.GRU if kind == "gru" else nn.LSTM
            self.body = recurrent(width, hidden, num_layers=layers, batch_first=True,
                                  bidirectional=(kind == "bilstm"),
                                  dropout=dropout if layers > 1 else 0.0)
            output = hidden * (2 if kind == "bilstm" else 1)
        elif kind == "cnn":
            stack, channels_in = [], width
            for _ in range(layers):
                stack += [nn.Conv1d(channels_in, hidden, kernel_size=3, padding=1), nn.ReLU()]
                channels_in = hidden
            self.body = nn.Sequential(*stack, nn.AdaptiveMaxPool1d(1))
            output = hidden
        else:
            self.project = nn.Linear(width, hidden)
            layer = nn.TransformerEncoderLayer(
                hidden, nhead=4, dim_feedforward=hidden * 2, dropout=dropout,
                batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, num_layers=layers)
            output = hidden
        self.head = nn.Sequential(
            nn.Linear(output + 1, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, codes, continuous, static):
        sequence = torch.cat([self.embedding(codes), continuous], dim=2)
        if self.kind in ("gru", "lstm", "bilstm"):
            pooled = self.body(sequence)[0][:, -1, :]
        elif self.kind == "cnn":
            pooled = self.body(sequence.transpose(1, 2)).squeeze(2)
        else:
            pooled = self.body(self.project(sequence)).mean(dim=1)
        return self.head(torch.cat([pooled, static], dim=1)).squeeze(1)


class Data:
    """The monthly matrix, the flat feature set and the splits, built once."""

    def __init__(self, seed: int = SEED):
        raw = load_taiwan()
        self.y_all = raw[TARGET].astype("int8")
        self.flat = engineer(raw).drop(columns=PROTECTED)

        limit = raw["LIMIT_BAL"].to_numpy(dtype=np.float32)
        self.codes = np.stack(
            [raw[f"PAY_{m}"].clip(-2, 8).to_numpy() + 2 for m in MONTHS], axis=1
        ).astype(np.int64)
        # Scaled by the customer's own limit, so the network sees behaviour rather than
        # account size; the limit itself goes in separately.
        self.continuous = np.clip(np.nan_to_num(np.stack([
            np.stack([raw[f"BILL_AMT{m}"].to_numpy(dtype=np.float32) / limit
                      for m in MONTHS], axis=1),
            np.stack([raw[f"PAY_AMT{m}"].to_numpy(dtype=np.float32) / limit
                      for m in MONTHS], axis=1),
        ], axis=2), nan=0.0, posinf=0.0, neginf=0.0), -3, 3)
        static = np.log1p(limit).reshape(-1, 1).astype(np.float32)

        train, rest = train_test_split(
            self.flat.index, test_size=0.30, stratify=self.y_all, random_state=seed)
        validation, test = train_test_split(
            rest, test_size=0.50, stratify=self.y_all.loc[rest], random_state=seed)
        self.index = {"train": train, "validation": validation, "test": test}
        self.rows = {name: self.flat.index.get_indexer(idx)
                     for name, idx in self.index.items()}
        self.y = {name: self.y_all.loc[idx].to_numpy(dtype=np.float32)
                  for name, idx in self.index.items()}
        self.static = (static - static[self.rows["train"]].mean()) / static[
            self.rows["train"]].std()

    def tensors(self, name: str):
        rows = self.rows[name]
        return (torch.from_numpy(self.codes[rows]), torch.from_numpy(self.continuous[rows]),
                torch.from_numpy(self.static[rows]))


def train_network(data: Data, kind: str, hidden: int, layers: int, dropout: float,
                  learning_rate: float, seed: int):
    """Fit on train, early stopping on validation AUC. Returns the best model and score."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SequenceModel(kind, hidden, layers, dropout)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_function = nn.BCEWithLogitsLoss()

    codes, continuous, static = data.tensors("train")
    target = torch.from_numpy(data.y["train"])
    validation = data.tensors("validation")

    best_auc, best_epoch, best_state, waited = 0.0, 0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(target))
        for start in range(0, len(order), BATCH):
            picked = order[start:start + BATCH]
            optimiser.zero_grad()
            loss_function(model(codes[picked], continuous[picked], static[picked]),
                          target[picked]).backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(*validation)).numpy()
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


def network_scores(model, data: Data, name: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(*data.tensors(name))).numpy()


def booster_scores(data: Data) -> dict:
    """The deployed configuration, fitted on train so both arms see the same data."""
    out = {}
    for name in ("validation", "test"):
        out[name] = np.mean([
            LGBMClassifier(**HYPERPARAMETERS, n_jobs=-1, random_state=seed, verbose=-1)
            .fit(data.flat.loc[data.index["train"]], data.y_all.loc[data.index["train"]])
            .predict_proba(data.flat.loc[data.index[name]])[:, 1] for seed in SEEDS], axis=0)
    return out


def ranked(values: np.ndarray) -> np.ndarray:
    return rankdata(values) / len(values)


def main() -> dict:
    torch.set_num_threads(4)
    data = Data()
    booster = booster_scores(data)
    booster_validation = float(roc_auc_score(data.y["validation"], booster["validation"]))
    booster_test = float(roc_auc_score(data.y["test"], booster["test"]))
    print(f"booster validation {booster_validation:.4f}  test {booster_test:.4f}", flush=True)

    search = []
    for kind in ARCHITECTURES:
        for configuration in CONFIGURATIONS:
            _, auc, epoch = train_network(data, kind, seed=SEED, **configuration)
            search.append({"architecture": kind, **configuration,
                           "validation_auc": auc, "epochs": epoch})
            print(f"  {kind:<10}{configuration} -> {auc:.4f}", flush=True)

    best = max(search, key=lambda row: row["validation_auc"])
    print(f"\nwinner: {best['architecture']} {best['validation_auc']:.4f}", flush=True)

    validation_scores, test_scores, seeds_used = [], [], []
    for seed in SEEDS:
        model, auc, epoch = train_network(
            data, best["architecture"], best["hidden"], best["layers"], best["dropout"],
            best["learning_rate"], seed)
        validation_scores.append(network_scores(model, data, "validation"))
        test_scores.append(network_scores(model, data, "test"))
        seeds_used.append({"seed": seed, "validation_auc": auc, "epochs": epoch})
        print(f"  seed {seed}: {auc:.4f}", flush=True)

    network = {"validation": np.mean(validation_scores, axis=0),
               "test": np.mean(test_scores, axis=0)}
    network_validation = float(roc_auc_score(data.y["validation"], network["validation"]))
    network_test = float(roc_auc_score(data.y["test"], network["test"]))

    # How differently do the two rank customers? A high correlation explains a small blend
    # gain: the engineered features and the raw sequence carry the same information.
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
        "booster": {"validation_auc": booster_validation, "test_auc": booster_test},
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

    out = artifacts_dir() / "sequence_models.json"
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
