"""Atari environment registration, preprocessing wrappers, and device selection."""

from __future__ import annotations

import os
from collections import deque

import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def data_dir(env_name: str) -> str:
    path = os.path.join(project_root(), "data", env_name)
    os.makedirs(path, exist_ok=True)
    return path


def register_atari_envs() -> bool:
    try:
        import ale_py

        gym.register_envs(ale_py)
        return True
    except ImportError:
        return False


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FrameSkip(gym.Wrapper):
    """Repeat action for k frames; max-pool last two observations."""

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip
        self._frames: deque = deque(maxlen=2)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._frames.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        if len(self._frames) == 1:
            frame = self._frames[0]
        else:
            frame = np.maximum(self._frames[-1], self._frames[-2])
        return frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames.clear()
        self._frames.append(obs)
        return obs, info


class AtariPreprocess(gym.Wrapper):
    """Grayscale resize + frame stack (channels-first for PyTorch convs)."""

    def __init__(
        self,
        env: gym.Env,
        frame_stack: int = 4,
        resize_shape: tuple[int, int] = (84, 84),
    ):
        super().__init__(env)
        self._frame_stack = frame_stack
        self._resize_shape = resize_shape
        self._frames: deque = deque(maxlen=frame_stack)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(frame_stack, resize_shape[0], resize_shape[1]),
            dtype=np.float32,
        )

    def _process(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, self._resize_shape, interpolation=cv2.INTER_AREA)
        return (resized / 255.0).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed = self._process(obs)
        self._frames.clear()
        for _ in range(self._frame_stack):
            self._frames.append(processed)
        return np.stack(self._frames, axis=0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(self._process(obs))
        return np.stack(self._frames, axis=0), reward, terminated, truncated, info


def make_atari_env(
    env_id: str,
    *,
    render_mode: str | None = None,
    frame_skip: int = 4,
    frame_stack: int = 4,
) -> gym.Env:
    base = gym.make(env_id, render_mode=render_mode, disable_env_checker=True)
    skipped = FrameSkip(base, skip=frame_skip)
    return AtariPreprocess(skipped, frame_stack=frame_stack)


def verify_rom(env_id: str) -> None:
    env = gym.make(env_id, render_mode=None, disable_env_checker=True)
    env.close()
