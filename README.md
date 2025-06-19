# Snake Game with Reinforcement Learning

Train AI agents to play Snake using deep reinforcement learning.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py [OPTIONS]
```

## Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-m`, `--mode` | `train` or `play` | `train` |
| `-s`, `--sessions` | Number of episodes/games | `1000` |
| `-mp`, `--model-path` | Path to model file (.pth) | `None` |
| `-np`, `--num-players` | Number of players (1-4) | `1` |
| `-ds`, `--display-speed` | Game speed (FPS) | `24` |
| `-r`, `--render` | Render every N episodes | `None` |
| `-se`, `--save-every` | Save model every N episodes | `100000` |
| `-sm`, `--smart-exploration` | Enable smart exploration | `False` |
| `-st`, `--step-by-step` | Step-by-step mode (play only) | `False` |

## Examples

**Train an agent:**
```bash
python main.py -m train -s 1000 -r 100
```

**Play with trained model:**
```bash
python main.py -m play -mp models/model.pth
```

**Multi-player training:**
```bash
python main.py -m train -np 4 -s 2000
```

**Step-by-step analysis:**
```bash
python main.py -m play -mp models/model.pth -st
```
