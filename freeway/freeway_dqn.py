"""Minimal DQN on ALE/Freeway-v5."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from datetime import datetime, timezone

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.atari_env import (
    AtariPreprocess,
    FrameSkip,
    data_dir,
    get_device,
    make_atari_env,
    register_atari_envs,
    verify_rom,
)
from common.duel_ipc import write_frame, write_metrics, write_status
from common.dqn import DQNAgent

ENV_ID = "ALE/Freeway-v5"
ENV_NAME = "freeway"
DEFAULT_EPISODES = 3000
CHECKPOINT = "freeway_dqn.pt"


def _log(msg: str) -> None:
    print(msg, flush=True)


def train(episodes: int = DEFAULT_EPISODES, render: bool = False, load_checkpoint: bool = False):
    device = get_device()
    _log(f"Device: {device}")
    _log(f"Starting training for {episodes} episodes (render={render})...")
    render_mode = "human" if render else None
    env = make_atari_env(ENV_ID, render_mode=render_mode)
    n_actions = env.action_space.n
    agent = DQNAgent(n_actions, device, lr=2.5e-4, epsilon_decay=0.9997)
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


def demo_eval(
    episodes: int = 5,
    checkpoint: str | None = None,
    duel_dir: str | None = None,
    max_steps_per_episode: int = 500,
    ipc_interval: int = 10,
    metrics_out: str | None = None,
    command: str | None = None,
) -> dict:
    """Evaluate with rgb_array frames and duel IPC for the dashboard."""
    device = get_device()
    duel_dir = duel_dir or os.path.join(data_dir(ENV_NAME), "duel")
    ckpt_path = checkpoint or os.path.join(data_dir(ENV_NAME), CHECKPOINT)
    cmd_str = command or " ".join(shlex.quote(a) for a in sys.argv)

    base = gym.make(ENV_ID, render_mode="rgb_array", disable_env_checker=True)
    skipped = FrameSkip(base, skip=4)
    env = AtariPreprocess(skipped, frame_stack=4)
    agent = DQNAgent(env.action_space.n, device)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    agent.load(ckpt_path)
    agent.epsilon = 0.0

    episode_rewards: list[float] = []
    write_status(
        duel_dir,
        episodes_total=episodes,
        episode=0,
        step=0,
        episode_reward=0.0,
        epsilon=agent.epsilon,
        checkpoint=ckpt_path,
        mean_reward_last_k=None,
        state="running",
        error=None,
    )

    try:
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            step = 0
            frame = base.render()
            if frame is not None:
                write_frame(duel_dir, frame)

            done = truncated = False
            while step < max_steps_per_episode and not (done or truncated):
                action = agent.select_action(state)
                state, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                step += 1

                if step % ipc_interval == 0 or done or truncated:
                    frame = base.render()
                    if frame is not None:
                        write_frame(duel_dir, frame)
                    last_k = episode_rewards[-10:] if episode_rewards else []
                    mean_k = (
                        float(np.mean(last_k + [episode_reward]))
                        if last_k or episode_reward
                        else None
                    )
                    write_status(
                        duel_dir,
                        episodes_total=episodes,
                        episode=episode + 1,
                        step=step,
                        episode_reward=episode_reward,
                        epsilon=agent.epsilon,
                        checkpoint=ckpt_path,
                        mean_reward_last_k=mean_k,
                        state="running",
                        error=None,
                    )

            episode_rewards.append(episode_reward)
            frame = base.render()
            if frame is not None:
                write_frame(duel_dir, frame)
            mean_k = float(np.mean(episode_rewards[-10:])) if episode_rewards else None
            write_status(
                duel_dir,
                episodes_total=episodes,
                episode=episode + 1,
                step=step,
                episode_reward=episode_reward,
                epsilon=agent.epsilon,
                checkpoint=ckpt_path,
                mean_reward_last_k=mean_k,
                state="running",
                error=None,
            )
    except Exception as exc:
        write_status(
            duel_dir,
            episodes_total=episodes,
            episode=len(episode_rewards),
            state="error",
            error=str(exc),
            checkpoint=ckpt_path,
        )
        env.close()
        base.close()
        raise

    env.close()
    base.close()
    rewards = np.array(episode_rewards, dtype=np.float64)
    metrics = {
        "repo": "gymnasium_two",
        "env": ENV_ID,
        "episodes": episodes,
        "mean_reward": float(rewards.mean()) if len(rewards) else 0.0,
        "std_reward": float(rewards.std()) if len(rewards) else 0.0,
        "min": float(rewards.min()) if len(rewards) else 0.0,
        "max": float(rewards.max()) if len(rewards) else 0.0,
        "checkpoint": ckpt_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": cmd_str,
    }
    write_metrics(duel_dir, metrics)
    if metrics_out:
        import json

        os.makedirs(os.path.dirname(metrics_out) or ".", exist_ok=True)
        with open(metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    write_status(
        duel_dir,
        episodes_total=episodes,
        episode=episodes,
        step=0,
        episode_reward=float(rewards[-1]) if len(rewards) else 0.0,
        epsilon=agent.epsilon,
        checkpoint=ckpt_path,
        mean_reward_last_k=float(rewards.mean()) if len(rewards) else None,
        state="done",
        error=None,
    )
    return metrics


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
    parser = argparse.ArgumentParser(description="DQN on ALE/Freeway-v5")
    parser.add_argument("--train", action="store_true", help="Train (skip prompts)")
    parser.add_argument("--eval", action="store_true", help="Evaluate a checkpoint")
    parser.add_argument("--demo-eval", action="store_true", help="Dashboard demo eval with IPC")
    parser.add_argument(
        "--dashboard-mode",
        action="store_true",
        help="Alias for --demo-eval (duel IPC, no GUI in this process)",
    )
    parser.add_argument("--render", action="store_true", help="Show game window")
    parser.add_argument("--load-checkpoint", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--metrics-out", type=str, default=None, help="Optional metrics JSON path")
    parser.add_argument(
        "--duel-dir",
        type=str,
        default=None,
        help="Duel IPC directory (default: data/freeway/duel/)",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=500,
        help="Cap steps per episode in demo-eval (default: 500)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=f"Episode count (default: {DEFAULT_EPISODES} train / 10 eval / 5 demo)",
    )
    return parser.parse_args()


def _cli_mode(args: argparse.Namespace) -> bool:
    return bool(
        args.train
        or args.eval
        or args.demo_eval
        or args.dashboard_mode
    )


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
    default_duel = os.path.join(data_dir(ENV_NAME), "duel")
    if args.duel_dir is None:
        args.duel_dir = default_duel

    if _cli_mode(args):
        if args.demo_eval or args.dashboard_mode:
            episodes = args.episodes if args.episodes is not None else 5
            demo_eval(
                episodes=episodes,
                checkpoint=args.checkpoint,
                duel_dir=args.duel_dir,
                max_steps_per_episode=args.max_steps_per_episode,
                metrics_out=args.metrics_out,
            )
            return
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
