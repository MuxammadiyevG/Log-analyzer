"""
LogTransformer — self-attention next-event predictor (Phase 4, Tier 3).

Same detection contract as DeepLog (next event outside top-k = violation), but
the sequence encoder is a Transformer instead of an LSTM. Self-attention can
look directly at any earlier position, so it captures long-range dependencies
(e.g. "CLOSE must match the OPEN seen many steps ago") that a bigram — and, past
its window, an LSTM — cannot. This is the small, CPU-trainable stand-in for the
transformer tier (LogBERT / NeuralLog family); the full LogLLM (BERT+Llama+QLoRA)
needs a GPU and is out of scope for this environment.

Unsupervised: trained on normal sessions only.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from app.models.deeplog import PAD, UNK  # shared reserved ids

__all__ = ["LogTransformerDetector", "PAD", "UNK"]


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):  # x: (batch, seq, d_model)
        return x + self.pe[:, : x.size(1)]


class _TransformerNet(nn.Module):
    def __init__(self, vocab: int, d_model: int, nhead: int, layers: int, ff: int, window: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.pos = _PositionalEncoding(d_model, window)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=ff, batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.fc = nn.Linear(d_model, vocab)

    def forward(self, x):  # x: (batch, window)
        h = self.pos(self.embed(x))
        h = self.encoder(h)
        return self.fc(h[:, -1, :])  # predict next event from last position


class LogTransformerDetector:
    """Transformer next-event-prediction detector over integer event sequences."""

    def __init__(self, vocab_size: int, config: Dict[str, Any] | None = None):
        config = config or {}
        self.vocab_size = int(vocab_size)
        self.window = int(config.get("window", 10))
        self.d_model = int(config.get("d_model", 64))
        self.nhead = int(config.get("nhead", 4))
        self.layers = int(config.get("num_layers", 2))
        self.ff = int(config.get("dim_feedforward", 128))
        self.epochs = int(config.get("epochs", 30))
        self.batch_size = int(config.get("batch_size", 256))
        self.lr = float(config.get("learning_rate", 1e-3))
        self.top_k = int(config.get("top_k", 9))

        self.device = torch.device(
            "cuda" if config.get("use_gpu") and torch.cuda.is_available() else "cpu"
        )
        self.model = _TransformerNet(
            self.vocab_size, self.d_model, self.nhead, self.layers, self.ff, self.window
        ).to(self.device)
        self.is_trained = False
        self.model_type = "log_transformer"

    def _windows(self, seq: Sequence[int]) -> List[Tuple[List[int], int]]:
        s = [PAD] * self.window + list(seq)
        return [(s[i : i + self.window], s[i + self.window]) for i in range(len(seq))]

    def fit(self, sequences: List[Sequence[int]]) -> "LogTransformerDetector":
        X, y = [], []
        for seq in sequences:
            for win, nxt in self._windows(seq):
                X.append(win)
                y.append(nxt)
        if not X:
            raise ValueError("No training windows produced")
        loader = DataLoader(
            TensorDataset(
                torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)
            ),
            batch_size=self.batch_size,
            shuffle=True,
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
                    f"LogTransformer epoch {epoch + 1}/{self.epochs} loss={total / len(loader):.4f}"
                )
        self.is_trained = True
        logger.info(f"LogTransformer trained: vocab={self.vocab_size}, window={self.window}")
        return self

    train = fit

    @torch.no_grad()
    def _violation_rate(self, seq: Sequence[int]) -> float:
        wins = self._windows(seq)
        if not wins:
            return 0.0
        X = torch.tensor([w for w, _ in wins], dtype=torch.long, device=self.device)
        targets = [t for _, t in wins]
        self.model.eval()
        logits = self.model(X)
        k = min(self.top_k, self.vocab_size)
        topk = torch.topk(logits, k, dim=1).indices.cpu().numpy()
        return sum(1 for i, t in enumerate(targets) if t not in topk[i]) / len(targets)

    def score(self, sequences: List[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        scores = np.array([self._violation_rate(s) for s in sequences], dtype=float)
        preds = (scores > 0.0).astype(int)
        return scores, preds

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "vocab_size": self.vocab_size,
                "window": self.window,
                "config": {
                    "d_model": self.d_model,
                    "nhead": self.nhead,
                    "num_layers": self.layers,
                    "dim_feedforward": self.ff,
                    "top_k": self.top_k,
                },
                "is_trained": self.is_trained,
            },
            path / "log_transformer.pt",
        )
        logger.info(f"LogTransformer saved to {path}")

    def load(self, path: Path) -> None:
        ckpt = torch.load(Path(path) / "log_transformer.pt", map_location=self.device)
        self.vocab_size = ckpt["vocab_size"]
        self.window = ckpt["window"]
        c = ckpt["config"]
        self.model = _TransformerNet(
            self.vocab_size, c["d_model"], c["nhead"], c["num_layers"], c["dim_feedforward"], self.window
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.top_k = c["top_k"]
        self.is_trained = ckpt["is_trained"]
        self.model.eval()
        logger.info(f"LogTransformer loaded from {path}")
