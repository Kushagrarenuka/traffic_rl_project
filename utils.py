from __future__ import annotations

import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def moving_average(values: List[float], window: int = 10) -> np.ndarray:
    if len(values) < window:
        return np.array(values, dtype=np.float32)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def save_training_plot(returns: List[float], path: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(returns, label="Episode return", alpha=0.35)
    smooth = moving_average(returns, window=10)
    if len(smooth) > 0:
        offset = len(returns) - len(smooth)
        plt.plot(range(offset, offset + len(smooth)), smooth, label="Moving average (10)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Training Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_csv(rows: List[Dict[str, float]], path: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
