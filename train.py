import argparse
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


class Cars:
    def __init__(self, max_episode_steps=2200, random_start=False):
        self.model = mujoco.MjModel.from_xml_path("./car.xml")
        self.data = mujoco.MjData(self.model)
        self.max_episode_steps = max_episode_steps
        self.random_start = random_start
        self.goal_xy = np.array([1.0, -1.0], dtype=np.float32)
        self.goal_radius = 0.16
        self.np_random = np.random.default_rng(0)
        self.step_count = 0
        self.prev_dist = None
        mujoco.mj_forward(self.model, self.data)

    @property
    def obs_dim(self):
        return 9

    @property
    def act_dim(self):
        return 2

    def _distance_to_goal(self):
        car_xy = self.data.body("car").xpos[:2]
        return float(np.linalg.norm(car_xy - self.goal_xy))

    def _forward_xy(self):
        return self.data.body("car").xmat.reshape(3, 3)[:, 0][:2]

    def _get_obs(self):
        car_xy = self.data.body("car").xpos[:2]
        qvel = self.data.qvel
        forward_xy = self._forward_xy()
        goal_delta = self.goal_xy - car_xy
        dist = np.linalg.norm(goal_delta)
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

    def reset(self, seed=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        if self.random_start:
            spawn_x = self.np_random.uniform(-0.5, 0.5)
            spawn_y = self.np_random.uniform(-0.5, 0.5)
            yaw = self.np_random.uniform(-np.pi, np.pi)
        else:
            spawn_x, spawn_y, yaw = 0.0, 0.0, 0.0

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
        action = np.asarray(action, dtype=np.float32)
        action = np.tanh(action)
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
            abs(self.data.body("car").xpos[0]) > 2.8
            or abs(self.data.body("car").xpos[1]) > 2.8
        )
        truncated = self.step_count >= self.max_episode_steps or out_of_bounds
        if out_of_bounds:
            reward -= 2.0

        obs = self._get_obs()
        info = {
            "distance_to_goal": dist,
            "success": float(terminated),
        }
        return obs, float(reward), terminated, truncated, info


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
def evaluate_policy(env, model, device, episodes=30):
    successes = 0
    returns = []
    final_distances = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        done = False
        ep_ret = 0.0
        info = {}
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mean_residual = model.policy_mean(model.policy_backbone(obs_t))[0].cpu().numpy()
            action = heuristic_action(obs) + 0.2 * mean_residual
            obs, r, term, trunc, info = env.step(action)
            ep_ret += r
            done = term or trunc
        successes += int(info.get("success", 0.0) > 0.5)
        returns.append(ep_ret)
        final_distances.append(info["distance_to_goal"])

    return {
        "success_rate": successes / episodes,
        "avg_return": float(np.mean(returns)),
        "avg_final_distance": float(np.mean(final_distances)),
    }


def save_learning_curve(
    episode_returns,
    eval_episodes,
    eval_success_rates,
    eval_final_distances,
    output_path,
):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    returns = np.array(episode_returns, dtype=np.float32)
    window = 50
    if len(returns) >= window:
        kernel = np.ones(window, dtype=np.float32) / window
        moving = np.convolve(returns, kernel, mode="valid")
        axes[0].plot(np.arange(window - 1, len(returns)), moving, label="Return (50-ep MA)")
    axes[0].plot(returns, alpha=0.2, label="Raw return")
    axes[0].set_title("Training Return")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode return")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(eval_episodes, eval_success_rates, label="Eval success rate")
    axes[1].plot(eval_episodes, eval_final_distances, label="Eval avg final distance")
    axes[1].set_title("Deterministic Evaluation (Actual Goal-Reaching)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Metric")
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def train(total_episodes=4000, max_episode_steps=2200, eval_every=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Cars(max_episode_steps=max_episode_steps, random_start=False)
    eval_env = Cars(max_episode_steps=max_episode_steps, random_start=False)

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = ActorCritic(env.obs_dim, env.act_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    gamma = 0.99
    entropy_coef = 0.002
    value_coef = 0.5

    episode_returns = []
    eval_episodes = []
    eval_success_rates = []
    eval_final_distances = []
    best_success = -1.0
    best_distance = np.inf
    best_state = None

    for episode in range(1, total_episodes + 1):
        obs, _ = env.reset(seed=episode)
        done = False
        rewards = []
        log_probs = []
        entropies = []
        values = []
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

            next_obs, reward, term, trunc, info = env.step(action.detach().cpu().numpy())
            done = term or trunc

            rewards.append(reward)
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            ep_return += reward
            obs = next_obs

        returns_t = discounted_returns(rewards, gamma=gamma, device=device)
        values_t = torch.stack(values)
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)

        advantages = returns_t - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(log_probs_t * advantages.detach()).mean()
        value_loss = F.mse_loss(values_t, returns_t)
        entropy_bonus = entropies_t.mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        episode_returns.append(ep_return)

        if episode % 50 == 0:
            avg50 = float(np.mean(episode_returns[-50:]))
            print(
                f"episode={episode:>4} avg_return(last50)={avg50:>8.3f} "
                f"last_dist={info['distance_to_goal']:.3f}"
            )

        if episode % eval_every == 0:
            metrics = evaluate_policy(eval_env, model, device, episodes=30)
            eval_episodes.append(episode)
            eval_success_rates.append(metrics["success_rate"])
            eval_final_distances.append(metrics["avg_final_distance"])
            print(
                f"[eval] ep={episode:>4} success_rate={metrics['success_rate']:.2f} "
                f"avg_final_dist={metrics['avg_final_distance']:.3f} "
                f"avg_return={metrics['avg_return']:.3f}"
            )
            improved = (
                metrics["success_rate"] > best_success
                or (
                    abs(metrics["success_rate"] - best_success) < 1e-8
                    and metrics["avg_final_distance"] < best_distance
                )
            )
            if improved:
                best_success = metrics["success_rate"]
                best_distance = metrics["avg_final_distance"]
                best_state = {
                    "policy_state_dict": model.state_dict(),
                    "obs_dim": env.obs_dim,
                    "action_dim": env.act_dim,
                    "success_rate": best_success,
                    "avg_final_distance": best_distance,
                }

    if best_state is None:
        best_state = {
            "policy_state_dict": model.state_dict(),
            "obs_dim": env.obs_dim,
            "action_dim": env.act_dim,
            "success_rate": 0.0,
            "avg_final_distance": float("inf"),
        }

    checkpoint_path = "car_reinforce_policy.pt"
    torch.save(best_state, checkpoint_path)
    plot_path = "car_learning_curve.png"
    save_learning_curve(
        episode_returns,
        eval_episodes,
        eval_success_rates,
        eval_final_distances,
        plot_path,
    )
    print(f"Saved best policy checkpoint to {checkpoint_path}")
    print(f"Saved learning curve plot to {plot_path}")
    print(
        f"Best eval success_rate={best_state['success_rate']:.2f}, "
        f"avg_final_distance={best_state['avg_final_distance']:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--max-episode-steps", type=int, default=2200)
    parser.add_argument("--eval-every", type=int, default=100)
    args = parser.parse_args()
    train(
        total_episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
        eval_every=args.eval_every,
    )
