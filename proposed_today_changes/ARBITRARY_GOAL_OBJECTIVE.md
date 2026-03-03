# New Objective: Arbitrary Goal Locations

## Goal
Train one policy that can drive to many different goal positions, not just one fixed sphere location.

## What I Added
- `train_arbitrary_goals.py`
  - Randomizes goal location every training episode
  - Moves the MuJoCo goal marker dynamically (`model.body_pos` for body `goal`)
  - Trains with actor-critic REINFORCE style updates
  - Evaluates on a grid of unseen/varied goal locations
  - Saves:
    - `car_arbitrary_goal_policy.pt`
    - `car_arbitrary_goal_curve.png`
    - `car_arbitrary_goal_eval.json`

- `main_arbitrary_goals.py`
  - Loads the arbitrary-goal policy
  - Runs MuJoCo viewer with either:
    - random new goals every reset, or
    - a fixed goal from CLI args

## Why This Helps
This changes the problem from memorizing one target to learning a goal-conditioned controller.  
Because the observation includes the goal delta vector, the same policy can generalize to many nearby goals.

## Run Commands
Train:
```bash
python3 train_arbitrary_goals.py --episodes 3000 --max-episode-steps 2200 --eval-every 100
```

Visualize with random goals:
```bash
python3 main_arbitrary_goals.py --policy car_arbitrary_goal_policy.pt
```

Visualize with a fixed goal:
```bash
python3 main_arbitrary_goals.py --policy car_arbitrary_goal_policy.pt --goal-x 1.2 --goal-y -0.4
```
