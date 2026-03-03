import argparse
import json
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class ArbitraryGoalCars:
    def __init__(self, max_episode_steps=2200):
        self.model = mujoco.MjModel.from_xml_path("./car.xml")
        self.data = mujoco.MjData(self.model)
        self.max_episode_steps = max_episode_steps
        self.goal_radius = 0.16
        self.goal_low = np.array([-1.8, -1.8], dtype=np.float32)
        self.goal_high = np.array([1.8, 1.8], dtype=np.float32)
        self.goal_xy = np.array([1.0, -1.0], dtype=np.float32)
        self.goal_body_id = self.model.body("goal").id
        self.step_count = 0
        self.prev_dist = None
        self.np_random = np.random.default_rng(0)
        self.set_goal(self.goal_xy)

    @property
    def obs_dim(self):
        return 9

    @property
    def act_dim(self):
        return 2

    def set_goal(self, goal_xy):
        self.goal_xy = np.asarray(goal_xy, dtype=np.float32)
        self.model.body_pos[self.goal_body_id, 0] = self.goal_xy[0]
        self.model.body_pos[self.goal_body_id, 1] = self.goal_xy[1]
        self.model.body_pos[self.goal_body_id, 2] = 0.03
        mujoco.mj_forward(self.model, self.data)

    def sample_goal(self):
        while True:
            goal = self.np_random.uniform(self.goal_low, self.goal_high)
            if np.linalg.norm(goal) > 0.45:
                return goal.astype(np.float32)

    def _distance_to_goal(self):
        car_xy = self.data.body("car").xpos[:2]
        return float(np.linalg.norm(car_xy - self.goal_xy))

    def _forward_xy(self):
        return self.data.body("car").xmat.reshape(3, 3)[:, 0][:2]

    def _get_obs(self):
        car_xy = self.data.body("car").xpos[:2]
        qvel = self.data.qvel
        goal_delta = self.goal_xy - car_xy
        forward_xy = self._forward_xy()
        return np.array(
            [
                car_xy[0],
                car_xy[1],
                qvel[0],
                qvel[1],
                qvel[5],
                forward_xy[0],
                forward_xy[1],
                goal_delta[0],
                goal_delta[1],
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, goal_xy=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        if goal_xy is None:
            goal_xy = self.sample_goal()
        self.set_goal(goal_xy)

        spawn_x = self.np_random.uniform(-0.25, 0.25)
        spawn_y = self.np_random.uniform(-0.25, 0.25)
        yaw = self.np_random.uniform(-np.pi, np.pi)
        self.data.qpos[0] = spawn_x
        self.data.qpos[1] = spawn_y
        self.data.qpos[2] = 0.03
        self.data.qpos[3] = np.cos(yaw / 2.0)
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = np.sin(yaw / 2.0)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.prev_dist = self._distance_to_goal()
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        action = np.tanh(np.asarray(action, dtype=np.float32))
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        dist = self._distance_to_goal()
        progress = self.prev_dist - dist
        self.prev_dist = dist

        goal_dir = self.goal_xy - self.data.body("car").xpos[:2]
        goal_norm = np.linalg.norm(goal_dir) + 1e-8
        align = float(np.dot(self._forward_xy(), goal_dir / goal_norm))

        reward = (
            6.0 * progress
            - 0.02 * dist
            + 0.1 * align
            - 0.01 * float(np.square(action[1]))
            - 0.003 * float(np.square(action[0]))
            - 0.001
        )

        terminated = dist < self.goal_radius
        if terminated:
            reward += 20.0

        out_of_bounds = (
            abs(self.data.body("car").xpos[0]) > 2.9
            or abs(self.data.body("car").xpos[1]) > 2.9
        )
        truncated = self.step_count >= self.max_episode_steps or out_of_bounds
        if out_of_bounds:
            reward -= 2.0

        info = {
            "distance_to_goal": dist,
            "success": float(terminated),
            "goal_x": float(self.goal_xy[0]),
            "goal_y": float(self.goal_xy[1]),
        }
        return self._get_obs(), float(reward), terminated, truncated, info


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.policy_backbone = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.value_backbone = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.policy_mean = nn.Linear(128, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.5))
        self.value_head = nn.Linear(128, 1)

    def policy_dist(self, x):
        feat = self.policy_backbone(x)
        mean = self.policy_mean(feat)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def value(self, x):
        feat = self.value_backbone(x)
        return self.value_head(feat).squeeze(-1)


def heuristic_action(obs):
    forward = obs[5:7]
    goal_delta = obs[7:9]
    dist = float(np.linalg.norm(goal_delta) + 1e-8)
    goal_dir = goal_delta / dist
    dot = float(np.clip(np.dot(forward, goal_dir), -1.0, 1.0))
    cross = float(forward[0] * goal_dir[1] - forward[1] * goal_dir[0])
    signed_angle = float(np.arctan2(cross, dot))
    turn = np.clip(2.5 * signed_angle, -1.0, 1.0)
    forward_cmd = np.clip(3.5 * dist * dot + 0.2 * np.sign(dot), -1.0, 1.0)
    return np.array([forward_cmd, turn], dtype=np.float32)


def discounted_returns(rewards, gamma, device):
    running = 0.0
    out = []
    for r in reversed(rewards):
        running = r + gamma * running
        out.insert(0, running)
    return torch.tensor(out, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate_generalization(env, model, device, goals):
    successes = 0
    distances = []
    returns = []

    for i, goal in enumerate(goals):
        obs, _ = env.reset(seed=20_000 + i, goal_xy=goal)
        done = False
        ep_return = 0.0
        info = {}
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            residual = model.policy_mean(model.policy_backbone(obs_t))[0].cpu().numpy()
            action = heuristic_action(obs) + 0.2 * residual
            obs, reward, term, trunc, info = env.step(action)
            ep_return += reward
            done = term or trunc
        successes += int(info.get("success", 0.0) > 0.5)
        distances.append(info["distance_to_goal"])
        returns.append(ep_return)

    return {
        "success_rate": successes / len(goals),
        "avg_final_distance": float(np.mean(distances)),
        "avg_return": float(np.mean(returns)),
    }


def build_eval_goals():
    grid_vals = [-1.5, -0.75, 0.0, 0.75, 1.5]
    goals = []
    for gx in grid_vals:
        for gy in grid_vals:
            goal = np.array([gx, gy], dtype=np.float32)
            if np.linalg.norm(goal) > 0.45:
                goals.append(goal)
    return goals


def save_curve(train_returns, eval_eps, eval_success, eval_dist, path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    returns = np.array(train_returns, dtype=np.float32)
    axes[0].plot(returns, alpha=0.2, label="Raw return")
    if len(returns) >= 50:
        kernel = np.ones(50, dtype=np.float32) / 50.0
        moving = np.convolve(returns, kernel, mode="valid")
        axes[0].plot(np.arange(49, len(returns)), moving, label="Return (50-ep MA)")
    axes[0].set_title("Arbitrary Goal Training Return")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(eval_eps, eval_success, label="Eval success rate")
    axes[1].plot(eval_eps, eval_dist, label="Eval avg final distance")
    axes[1].set_title("Generalization Evaluation (Many Goal Locations)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Metric")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def train(total_episodes=5000, max_episode_steps=2200, eval_every=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    env = ArbitraryGoalCars(max_episode_steps=max_episode_steps)
    eval_env = ArbitraryGoalCars(max_episode_steps=max_episode_steps)
    model = ActorCritic(env.obs_dim, env.act_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    gamma = 0.99
    entropy_coef = 0.002
    value_coef = 0.5
    eval_goals = build_eval_goals()

    train_returns = []
    eval_eps = []
    eval_success = []
    eval_dist = []
    best_success = -1.0
    best_dist = np.inf
    best_ckpt = None

    for episode in range(1, total_episodes + 1):
        obs, _ = env.reset(seed=episode)
        done = False
        rewards = []
        log_probs = []
        values = []
        entropies = []
        ep_return = 0.0
        info = {}

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            dist = model.policy_dist(obs_t)
            residual = dist.sample()[0]
            log_prob = dist.log_prob(residual).sum()
            entropy = dist.entropy().sum()
            value = model.value(obs_t)[0]
            action = torch.tensor(heuristic_action(obs), dtype=torch.float32, device=device) + 0.2 * residual

            obs, reward, term, trunc, info = env.step(action.detach().cpu().numpy())
            done = term or trunc

            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)
            ep_return += reward

        returns_t = discounted_returns(rewards, gamma=gamma, device=device)
        values_t = torch.stack(values)
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)
        advantages = returns_t - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(log_probs_t * advantages.detach()).mean()
        value_loss = F.mse_loss(values_t, returns_t)
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropies_t.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_returns.append(ep_return)

        if episode % 50 == 0:
            print(
                f"episode={episode:>4} avg_return(last50)={np.mean(train_returns[-50:]):>8.3f} "
                f"last_goal=({info['goal_x']:.2f},{info['goal_y']:.2f})"
            )

        if episode % eval_every == 0:
            metrics = evaluate_generalization(eval_env, model, device, eval_goals)
            eval_eps.append(episode)
            eval_success.append(metrics["success_rate"])
            eval_dist.append(metrics["avg_final_distance"])
            print(
                f"[eval] ep={episode:>4} success_rate={metrics['success_rate']:.2f} "
                f"avg_final_dist={metrics['avg_final_distance']:.3f}"
            )
            improved = (
                metrics["success_rate"] > best_success
                or (
                    abs(metrics["success_rate"] - best_success) < 1e-8
                    and metrics["avg_final_distance"] < best_dist
                )
            )
            if improved:
                best_success = metrics["success_rate"]
                best_dist = metrics["avg_final_distance"]
                best_ckpt = {
                    "policy_state_dict": model.state_dict(),
                    "obs_dim": env.obs_dim,
                    "action_dim": env.act_dim,
                    "success_rate": best_success,
                    "avg_final_distance": best_dist,
                    "goal_mode": "arbitrary",
                }

    if best_ckpt is None:
        best_ckpt = {
            "policy_state_dict": model.state_dict(),
            "obs_dim": env.obs_dim,
            "action_dim": env.act_dim,
            "success_rate": 0.0,
            "avg_final_distance": float("inf"),
            "goal_mode": "arbitrary",
        }

    ckpt_path = "car_arbitrary_goal_policy.pt"
    curve_path = "car_arbitrary_goal_curve.png"
    summary_path = "car_arbitrary_goal_eval.json"

    torch.save(best_ckpt, ckpt_path)
    save_curve(train_returns, eval_eps, eval_success, eval_dist, curve_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_eval_success_rate": best_ckpt["success_rate"],
                "best_eval_avg_final_distance": best_ckpt["avg_final_distance"],
                "eval_goals_count": len(eval_goals),
            },
            f,
            indent=2,
        )

    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved curve: {curve_path}")
    print(f"Saved eval summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max-episode-steps", type=int, default=2200)
    parser.add_argument("--eval-every", type=int, default=100)
    args = parser.parse_args()
    train(
        total_episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
        eval_every=args.eval_every,
    )
