# GymnasiumTwo

A clean workspace for new [Gymnasium](https://gymnasium.farama.org/) reinforcement-learning experiments, isolated from the [Gynasium](https://github.com/) repo and its `GymnasiumVENV`.

## Environments

| Folder | Gymnasium ID | Notes |
|--------|----------------|-------|
| `pong/` | `ALE/Pong-v5` | Classic two-player paddle game; good DQN/PPO baseline |
| `freeway/` | `ALE/Freeway-v5` | Simple sparse-reward game; chicken crosses traffic |

Both use standard Atari preprocessing (frame skip, grayscale 84×84, 4-frame stack) via `common/atari_env.py`.

## Setup

```bash
cd "/path/to/GymnasiumTwo"
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

If Atari ROMs are missing, install the ROM package and accept the license when prompted:

```bash
pip install autorom[accept-rom-license]
AutoROM --accept-license
```

Select **`.venv/bin/python`** as the interpreter for this folder in Cursor/VS Code (separate from Gynasium’s `GymnasiumVENV`).

## Run

```bash
source .venv/bin/activate
python pong/pong_dqn.py --train
python freeway/freeway_dqn.py --train
```

Or run interactively (no flags) and answer the prompts. CLI flags:

- `--train` / `--eval`
- `--render` — show the game window
- `--load-checkpoint` — resume training
- `--episodes N` — override episode count

Checkpoints and plots go under `data/pong/` and `data/freeway/` (gitignored).

## Layout

```
GymnasiumTwo/
├── .venv/              # local only
├── common/             # shared Atari env + preprocessing
├── pong/
├── freeway/
├── data/               # models, logs (gitignored)
└── requirements.txt
```

## Multi-root workspace

Open **`GymnasiumAgents/GymnasiumAgents.code-workspace`** (recommended) to work across the hub, Gynasium, and GymnasiumTwo. Use a different Python interpreter per root (`.venv` here vs `Gynasium/GymnasiumVENV`).
