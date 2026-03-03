import argparse
import datetime as dt
import json
import os
import subprocess

import numpy as np
import torch

from train import ActorCritic as FixedModel
from train import Cars as FixedCars
from train import heuristic_action as fixed_heuristic_action
from train_arbitrary_goals import ActorCritic as ArbitraryModel
from train_arbitrary_goals import ArbitraryGoalCars
from train_arbitrary_goals import build_eval_goals
from train_arbitrary_goals import heuristic_action as arbitrary_heuristic_action


def git_commit_short():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


def load_fixed_policy(path):
    checkpoint = torch.load(path, map_location="cpu")
    model = FixedModel(obs_dim=checkpoint["obs_dim"], act_dim=checkpoint["action_dim"])
    model.load_state_dict(checkpoint["policy_state_dict"])
    model.eval()
    return model


def load_arbitrary_policy(path):
    checkpoint = torch.load(path, map_location="cpu")
    model = ArbitraryModel(obs_dim=checkpoint["obs_dim"], act_dim=checkpoint["action_dim"])
    model.load_state_dict(checkpoint["policy_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_fixed(model, episodes=30, max_episode_steps=2200):
    env = FixedCars(max_episode_steps=max_episode_steps, random_start=True)
    success = 0
    final_distances = []
    returns = []
    lengths = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        ep_return = 0.0
        steps = 0
        info = {}

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            residual = model.policy_mean(model.policy_backbone(obs_t))[0].numpy()
            action = fixed_heuristic_action(obs) + 0.2 * residual
            obs, reward, term, trunc, info = env.step(action)
            ep_return += reward
            steps += 1
            done = term or trunc

        success += int(info.get("success", 0.0) > 0.5)
        final_distances.append(info["distance_to_goal"])
        returns.append(ep_return)
        lengths.append(steps)

    return {
        "episodes": episodes,
        "success_rate": success / episodes,
        "avg_final_distance": float(np.mean(final_distances)),
        "avg_return": float(np.mean(returns)),
        "avg_episode_length": float(np.mean(lengths)),
    }


@torch.no_grad()
def evaluate_arbitrary(model, max_episode_steps=2200):
    env = ArbitraryGoalCars(max_episode_steps=max_episode_steps)
    goals = build_eval_goals()
    success = 0
    final_distances = []
    returns = []
    lengths = []

    for i, goal in enumerate(goals):
        obs, _ = env.reset(seed=2000 + i, goal_xy=goal)
        done = False
        ep_return = 0.0
        steps = 0
        info = {}

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            residual = model.policy_mean(model.policy_backbone(obs_t))[0].numpy()
            action = arbitrary_heuristic_action(obs) + 0.2 * residual
            obs, reward, term, trunc, info = env.step(action)
            ep_return += reward
            steps += 1
            done = term or trunc

        success += int(info.get("success", 0.0) > 0.5)
        final_distances.append(info["distance_to_goal"])
        returns.append(ep_return)
        lengths.append(steps)

    return {
        "episodes": len(goals),
        "success_rate": success / len(goals),
        "avg_final_distance": float(np.mean(final_distances)),
        "avg_return": float(np.mean(returns)),
        "avg_episode_length": float(np.mean(lengths)),
    }


def write_markdown(path, metrics):
    fixed = metrics["fixed_goal"]
    arbitrary = metrics["arbitrary_goal_grid"]
    lines = [
        "# Latest Evaluation Metrics",
        "",
        f"- Generated UTC: `{metrics['generated_utc']}`",
        f"- Commit: `{metrics['git_commit']}`",
        "",
        "| Scenario | Episodes | Success Rate | Avg Final Distance | Avg Return | Avg Episode Length |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Fixed Goal (random starts) | {fixed['episodes']} | {fixed['success_rate']:.3f} | "
            f"{fixed['avg_final_distance']:.3f} | {fixed['avg_return']:.3f} | {fixed['avg_episode_length']:.1f} |"
        ),
        (
            f"| Arbitrary Goals (grid) | {arbitrary['episodes']} | {arbitrary['success_rate']:.3f} | "
            f"{arbitrary['avg_final_distance']:.3f} | {arbitrary['avg_return']:.3f} | {arbitrary['avg_episode_length']:.1f} |"
        ),
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-policy", type=str, default="car_reinforce_policy.pt")
    parser.add_argument("--arbitrary-policy", type=str, default="car_arbitrary_goal_policy.pt")
    parser.add_argument("--fixed-episodes", type=int, default=30)
    parser.add_argument("--max-episode-steps", type=int, default=2200)
    parser.add_argument("--out-json", type=str, default="results/latest_metrics.json")
    parser.add_argument("--out-md", type=str, default="results/latest_metrics.md")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)

    fixed_model = load_fixed_policy(args.fixed_policy)
    arbitrary_model = load_arbitrary_policy(args.arbitrary_policy)

    fixed_metrics = evaluate_fixed(
        fixed_model,
        episodes=args.fixed_episodes,
        max_episode_steps=args.max_episode_steps,
    )
    arbitrary_metrics = evaluate_arbitrary(
        arbitrary_model,
        max_episode_steps=args.max_episode_steps,
    )

    metrics = {
        "generated_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "git_commit": git_commit_short(),
        "fixed_goal": fixed_metrics,
        "arbitrary_goal_grid": arbitrary_metrics,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    write_markdown(args.out_md, metrics)

    print(f"Saved metrics json: {args.out_json}")
    print(f"Saved metrics markdown: {args.out_md}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
