# RL + MuJoCo Learning (Car to Goal Sphere)

## Project Overview
I built this project to train a MuJoCo car agent to reach a target sphere using reinforcement learning.

The environment is defined in `car.xml`, training happens in `train.py`, and visualization/inference happens in `main.py`.

## Prerequisites (Install First)

### System
- Python 3.10+ (I tested on Python 3.12)
- `pip3`

### Python Packages
I install dependencies using:

```bash
pip3 install -r requirements.txt
```

If you want to install manually instead:

```bash
pip3 install numpy matplotlib torch mujoco
```

### Optional (recommended) virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

## What Was Wrong in the Original Version
When I started from the original code, the behavior looked random in simulation and the car often did not actually reach the sphere even when the training curve looked like it was improving.

Main issues in the original setup:
- The training loop was incomplete/stub-like.
- The learning signal and episode handling were not aligned well enough with actual goal reaching.
- The training graph could look better even when real sphere-reaching behavior was still poor.
- Inference and training logic were not tightly aligned.

## What I Changed

### 1) Rebuilt training into a complete RL pipeline
In `train.py`, I replaced the old flow with a full actor-critic style REINFORCE training loop:
- collect episode trajectories
- compute discounted returns
- compute advantages with a value baseline
- apply policy/value optimization each episode

This made learning stable and actually trainable.

### 2) Made the task explicitly goal-conditioned
I changed observations to include features that directly matter for navigation:
- car position/velocity
- forward direction
- vector from car to goal

This gave the model direct awareness of where the sphere is relative to the car.

### 3) Fixed reward and termination around real task success
I redesigned reward shaping so it reflects true progress:
- positive reward for reducing distance to goal
- alignment reward for facing toward the goal
- control penalties to reduce unstable spinning
- success bonus when entering goal radius

Episode termination now clearly captures:
- success (inside goal radius)
- truncation (time limit or out-of-bounds)

### 4) Added deterministic evaluation metrics (real behavior checks)
To avoid fake-looking learning curves, I added periodic deterministic evaluation:
- `success_rate` (fraction of eval episodes that reach the sphere)
- `avg_final_distance` (how close the car ends on average)
- `avg_return`

Most importantly, I now save the **best checkpoint by true success**, not just raw training return.

### 5) Improved control with heuristic + learned residual
I added a goal-directed heuristic controller and trained the network as a residual policy on top of it.

Final control = heuristic command + small learned correction

This removed random wandering and made the agent reliably drive to the sphere.

### 6) Synced inference with training behavior
In `main.py`, I updated the runtime controller to use the same observation format and control logic as training.

That means what I train is what I actually run in MuJoCo.

### 7) Improved learning curve output
I updated the graph generation so `car_learning_curve.png` now includes:
- training return trend
- evaluation success rate
- evaluation average final distance

This gives a realistic view of whether the agent is truly learning to reach the sphere.

### 8) Headless plotting reliability fix
I set Matplotlib to `Agg` backend so plotting works reliably in non-GUI/headless runs.

## Files and Roles
- `car.xml`: MuJoCo model and world setup
- `train.py`: training + evaluation + checkpoint + learning curve generation
- `main.py`: model loading + MuJoCo viewer rollout
- `car_reinforce_policy.pt`: trained checkpoint (generated)
- `car_learning_curve.png`: learning/eval graph (generated)

## How To Run

### Train
```bash
python3 train.py --episodes 300 --max-episode-steps 2200 --eval-every 50
```

### Visualize in MuJoCo
```bash
python3 main.py
```

## Outputs
After training, I expect:
- `car_reinforce_policy.pt` to be updated
- `car_learning_curve.png` to be updated
- console logs that include eval metrics like `success_rate` and `avg_final_dist`

## Notes
This version is focused on making behavior in simulation match what the metrics claim. The main goal of these updates was to make learning curves honest and make the car consistently reach the target sphere.
