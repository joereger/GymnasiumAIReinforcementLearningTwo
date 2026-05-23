"""IPC files for Freeway duel dashboard (GymnasiumTwo side)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

REPO_ID = "gymnasium_two"

LIVE_STATUS = "live_status.json"
LATEST_FRAME = "latest_frame.png"
METRICS = "metrics.json"


def ensure_duel_dir(duel_dir: str) -> None:
    os.makedirs(duel_dir, exist_ok=True)


def write_status(duel_dir: str, **fields: Any) -> None:
    ensure_duel_dir(duel_dir)
    payload: dict[str, Any] = {
        "repo": REPO_ID,
        "episode": 0,
        "episodes_total": 0,
        "step": 0,
        "episode_reward": 0.0,
        "epsilon": 0.0,
        "checkpoint": "",
        "mean_reward_last_k": None,
        "state": "running",
        "error": None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(fields)
    payload["repo"] = REPO_ID
    path = os.path.join(duel_dir, LIVE_STATUS)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_frame(duel_dir: str, rgb: np.ndarray) -> None:
    ensure_duel_dir(duel_dir)
    frame = np.asarray(rgb)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    path = os.path.join(duel_dir, LATEST_FRAME)
    cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def write_metrics(duel_dir: str, metrics: dict[str, Any]) -> None:
    ensure_duel_dir(duel_dir)
    metrics = dict(metrics)
    metrics.setdefault("repo", REPO_ID)
    metrics.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    path = os.path.join(duel_dir, METRICS)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
