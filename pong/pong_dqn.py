"""Minimal DQN on ALE/Pong-v5."""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.atari_env import (
    data_dir,
    get_device,
    make_atari_env,
    register_atari_envs,
    verify_rom,
)
from common.dqn import DQNAgent

ENV_ID = "ALE/Pong-v5"
ENV_NAME = "pong"
DEFAULT_EPISODES = 2000
CHECKPOINT = "pong_dqn.pt"


def _log(msg: str) -> None:
    print(msg, flush=True)


def train(episodes: int = DEFAULT_EPISODES, render: bool = False, load_checkpoint: bool = False):
    device = get_device()
    _log(f"Device: {device}")
    _log(f"Starting training for {episodes} episodes (render={render})...")
    render_mode = "human" if render else None
    env = make_atari_env(ENV_ID, render_mode=render_mode)
    n_actions = env.action_space.n
    agent = DQNAgent(n_actions, device)
    ckpt_path = os.path.join(data_dir(ENV_NAME), CHECKPOINT)

    if load_checkpoint and os.path.isfile(ckpt_path):
        agent.load(ckpt_path)
        _log(f"Loaded checkpoint: {ckpt_path}")

    rewards_history = []
    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.remember(state, action, reward, next_state, float(done or truncated))
            agent.learn()
            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)
        avg100 = np.mean(rewards_history[-100:])
        _log(
            f"Episode {episode}/{episodes}  reward={total_reward:.1f}  "
            f"avg100={avg100:.1f}  eps={agent.epsilon:.3f}"
        )

        if episode % 100 == 0:
            agent.save(ckpt_path)

    agent.save(ckpt_path)
    env.close()
    _plot_rewards(rewards_history, ENV_NAME)
    return agent


def evaluate(episodes: int = 10, render: bool = True):
    device = get_device()
    render_mode = "human" if render else None
    env = make_atari_env(ENV_ID, render_mode=render_mode)
    agent = DQNAgent(env.action_space.n, device)
    ckpt_path = os.path.join(data_dir(ENV_NAME), CHECKPOINT)
    if not os.path.isfile(ckpt_path):
        _log(f"No checkpoint at {ckpt_path}. Train first.")
        env.close()
        return
    agent.load(ckpt_path)
    agent.epsilon = 0.0

    scores = []
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        total = 0.0
        done = truncated = False
        while not (done or truncated):
            action = agent.select_action(state)
            state, reward, done, truncated, _ = env.step(action)
            total += reward
        scores.append(total)
        _log(f"Eval episode {ep}: reward={total:.1f}")

    _log(f"Mean reward: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    env.close()


def _plot_rewards(rewards: list[float], env_name: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.4, label="episode")
    if len(rewards) >= 10:
        kernel = min(100, len(rewards))
        rolling = np.convolve(rewards, np.ones(kernel) / kernel, mode="valid")
        plt.plot(range(kernel - 1, len(rewards)), rolling, label=f"{kernel}-ep avg")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{env_name} DQN training")
    plt.legend()
    out = os.path.join(data_dir(env_name), f"{env_name}_training.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    _log(f"Saved plot: {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQN on ALE/Pong-v5")
    parser.add_argument("--train", action="store_true", help="Train (skip prompts)")
    parser.add_argument("--eval", action="store_true", help="Evaluate a checkpoint")
    parser.add_argument("--render", action="store_true", help="Show game window")
    parser.add_argument("--load-checkpoint", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=f"Episode count (default: {DEFAULT_EPISODES} train / 10 eval)",
    )
    return parser.parse_args()


def main():
    if not register_atari_envs():
        _log("Install ale-py: pip install ale-py")
        sys.exit(1)
    try:
        verify_rom(ENV_ID)
        _log(f"{ENV_ID} ROM OK.")
    except Exception as exc:
        _log(f"ROM check failed: {exc}")
        _log("Try: pip install autorom && AutoROM --accept-license")
        sys.exit(1)

    args = parse_args()
    if args.train:
        episodes = args.episodes if args.episodes is not None else DEFAULT_EPISODES
        train(episodes=episodes, render=args.render, load_checkpoint=args.load_checkpoint)
        return
    if args.eval:
        episodes = args.episodes if args.episodes is not None else 10
        evaluate(episodes=episodes, render=args.render)
        return

    while True:
        choice = input("Train or evaluate? [t/e]: ").strip().lower()
        if choice in ("t", "train"):
            render = input("Render during training? [y/n]: ").strip().lower() in ("y", "yes")
            load_ckpt = input("Load checkpoint? [y/n]: ").strip().lower() in ("y", "yes")
            train(render=render, load_checkpoint=load_ckpt)
            break
        if choice in ("e", "eval", "evaluate"):
            render = input("Render? [y/n]: ").strip().lower() in ("y", "yes")
            evaluate(render=render)
            break
        _log("Enter 't' or 'e'.")


if __name__ == "__main__":
    main()
