from __future__ import annotations

import csv
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_demand_profile(length: int) -> np.ndarray:
    """
    Ramp-up / ramp-down demand profile peaking at the midpoint.
    Centralised here to avoid duplication — was copy-pasted in the
    original train.py and evaluate.py.
    """
    x = np.linspace(0.8, 1.4, length // 2)
    y = np.linspace(1.4, 0.9, length - len(x))
    return np.concatenate([x, y]).astype(np.float32)


def moving_average(values: List[float], window: int = 10) -> np.ndarray:
    if len(values) < window:
        return np.array(values, dtype=np.float32)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def save_training_plot(returns: List[float], path: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(returns, alpha=0.35, label="Episode return")
    smooth = moving_average(returns, window=10)
    if len(smooth):
        offset = len(returns) - len(smooth)
        plt.plot(range(offset, offset + len(smooth)), smooth, label="Moving avg (10)")
    plt.xlabel("Episode"); plt.ylabel("Return"); plt.title("Training Performance")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def save_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)