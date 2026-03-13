# Genetic Dodge Runner

A Dodge Runner game prototype built with an Entity-Component-System (ECS) architecture, designed for Reinforcement Learning training and manual play.

## Project Structure

```text
/
├── src/
│   ├── core/           # ECS Physics Engine
│   ├── rl/             # RL Wrapper (Rewards/Observations)
│   └── ui/             # Renderer and State Machine
├── config.py           # Game configuration
├── entities.py         # Action enums
└── main.py             # Entry point
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Controls

- Move: WASD or Arrow Keys
- Restart: R
- Quit: ESC
