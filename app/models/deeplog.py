"""
DeepLog — LSTM next-event-prediction sequence anomaly detector (Phase 3).

The point-wise models (Phase 1/2) score each line in isolation, so they are
blind to anomalies that live in *order and rate*: a brute-force burst is a
stream of individually-normal `POST /login 401` lines; a scan is a normal-looking
sequence of requests in an abnormal order. DeepLog (Du et al., CCS 2017) catches
these.

Idea: learn the grammar of normal event sequences. Train an LSTM to predict the
next event id from a sliding window of previous event ids. At inference, if the
event that actually occurred is not among the model's top-k predictions, that
step violates the learned grammar → anomalous. A session is anomalous if it
contains any violation (the original DeepLog rule); we also expose a continuous
violation-rate score so the harness can sweep thresholds.

Unsupervised: trained on normal sessions only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

PAD = 0  # reserved: left-padding for short sessions
UNK = 1  # reserved: event unseen during training


class _DeepLogNet(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden: int, layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(
            embedding_dim, hidden, num_layers=layers, batch_first=True
        )
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):  # x: (batch, window) int64
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        return self.fc(out[:, -1, :])  # logits over vocab for the next event


class DeepLogDetector:
    """LSTM next-event-prediction detector over integer event sequences."""

    def __init__(self, vocab_size: int, config: Dict[str, Any] | None = None):
        config = config or {}
        self.vocab_size = int(vocab_size)
        self.window = int(config.get("window", 10))
        self.embedding_dim = int(config.get("embedding_dim", 32))
        self.hidden = int(config.get("hidden", 64))
        self.layers = int(config.get("num_layers", 2))
        self.epochs = int(config.get("epochs", 30))
        self.batch_size = int(config.get("batch_size", 256))
        self.lr = float(config.get("learning_rate", 1e-3))
        self.top_k = int(config.get("top_k", 9))

        self.device = torch.device(
            "cuda" if config.get("use_gpu") and torch.cuda.is_available() else "cpu"
        )
        self.model = _DeepLogNet(
            self.vocab_size, self.embedding_dim, self.hidden, self.layers
        ).to(self.device)
        self.is_trained = False
        self.model_type = "deeplog"

    # -- windowing ----------------------------------------------------------

    def _windows(self, seq: Sequence[int]) -> List[Tuple[List[int], int]]:
        """Left-pad and slide: (window of W events) -> next event."""
        s = [PAD] * self.window + list(seq)
        return [(s[i : i + self.window], s[i + self.window]) for i in range(len(seq))]

    def _build_dataset(self, sequences: List[Sequence[int]]):
        X, y = [], []
        for seq in sequences:
            for win, nxt in self._windows(seq):
                X.append(win)
                y.append(nxt)
        if not X:
            raise ValueError("No training windows produced (empty sequences)")
        return (
            torch.tensor(X, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )

    # -- fit / score --------------------------------------------------------

    def fit(self, sequences: List[Sequence[int]]) -> "DeepLogDetector":
        X, y = self._build_dataset(sequences)
        loader = DataLoader(
            TensorDataset(X, y), batch_size=self.batch_size, shuffle=True
        )
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        crit = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            total = 0.0
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                opt.zero_grad()
                loss = crit(self.model(bx), by)
                loss.backward()
                opt.step()
                total += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"DeepLog epoch {epoch + 1}/{self.epochs} loss={total / len(loader):.4f}"
                )
        self.is_trained = True
        logger.info(f"DeepLog trained: vocab={self.vocab_size}, window={self.window}")
        return self

    train = fit

    @torch.no_grad()
    def _violation_rate(self, seq: Sequence[int]) -> float:
        """Fraction of steps where the true next event is outside top-k."""
        wins = self._windows(seq)
        if not wins:
            return 0.0
        X = torch.tensor([w for w, _ in wins], dtype=torch.long, device=self.device)
        targets = [t for _, t in wins]
        self.model.eval()
        logits = self.model(X)
        k = min(self.top_k, self.vocab_size)
        topk = torch.topk(logits, k, dim=1).indices.cpu().numpy()
        violations = sum(
            1 for i, t in enumerate(targets) if t not in topk[i]
        )
        return violations / len(targets)

    def score(
        self, sequences: List[Sequence[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (scores, predictions) per sequence. Score = violation rate."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        scores = np.array([self._violation_rate(s) for s in sequences], dtype=float)
        preds = (scores > 0.0).astype(int)  # DeepLog rule: any violation = anomaly
        return scores, preds

    # -- persistence --------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "vocab_size": self.vocab_size,
                "window": self.window,
                "config": {
                    "embedding_dim": self.embedding_dim,
                    "hidden": self.hidden,
                    "num_layers": self.layers,
                    "top_k": self.top_k,
                },
                "is_trained": self.is_trained,
            },
            path / "deeplog.pt",
        )
        logger.info(f"DeepLog saved to {path}")

    def load(self, path: Path) -> None:
        ckpt = torch.load(Path(path) / "deeplog.pt", map_location=self.device)
        self.vocab_size = ckpt["vocab_size"]
        self.window = ckpt["window"]
        c = ckpt["config"]
        self.model = _DeepLogNet(
            self.vocab_size, c["embedding_dim"], c["hidden"], c["num_layers"]
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.top_k = c["top_k"]
        self.is_trained = ckpt["is_trained"]
        self.model.eval()
        logger.info(f"DeepLog loaded from {path}")
