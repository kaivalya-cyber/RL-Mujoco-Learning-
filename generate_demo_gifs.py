import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch

from train import ActorCritic as FixedModel
from train import Cars as FixedCars
from train import heuristic_action as fixed_heuristic_action
from train_arbitrary_goals import ActorCritic as ArbitraryModel
from train_arbitrary_goals import ArbitraryGoalCars
from train_arbitrary_goals import heuristic_action as arbitrary_heuristic_action


def load_policy(path, model_cls):
    checkpoint = torch.load(path, map_location="cpu")
    model = model_cls(obs_dim=checkpoint["obs_dim"], act_dim=checkpoint["action_dim"])
    model.load_state_dict(checkpoint["policy_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def rollout_fixed(policy_path, max_steps=2200):
    model = load_policy(policy_path, FixedModel)
    env = FixedCars(max_episode_steps=max_steps, random_start=False)
    obs, _ = env.reset(seed=123)

    traj = []
    for _ in range(max_steps):
        car_xy = env.data.body("car").xpos[:2].copy()
        fwd = env.data.body("car").xmat.reshape(3, 3)[:, 0][:2].copy()
        traj.append(
            {
                "car_xy": car_xy,
                "goal_xy": env.goal_xy.copy(),
                "fwd_xy": fwd,
            }
        )

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        residual = model.policy_mean(model.policy_backbone(obs_t))[0].numpy()
        action = fixed_heuristic_action(obs) + 0.2 * residual
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    return traj


@torch.no_grad()
def rollout_arbitrary(policy_path, goal_xy=None, max_steps=2200):
    model = load_policy(policy_path, ArbitraryModel)
    env = ArbitraryGoalCars(max_episode_steps=max_steps)
    obs, _ = env.reset(seed=456, goal_xy=goal_xy)

    traj = []
    for _ in range(max_steps):
        car_xy = env.data.body("car").xpos[:2].copy()
        fwd = env.data.body("car").xmat.reshape(3, 3)[:, 0][:2].copy()
        traj.append(
            {
                "car_xy": car_xy,
                "goal_xy": env.goal_xy.copy(),
                "fwd_xy": fwd,
            }
        )

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        residual = model.policy_mean(model.policy_backbone(obs_t))[0].numpy()
        action = arbitrary_heuristic_action(obs) + 0.2 * residual
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    return traj


def save_trajectory_gif(traj, out_path, title, frame_stride=6):
    sampled = traj[::frame_stride] if frame_stride > 1 else traj
    if len(sampled) < 2:
        sampled = traj

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    (path_line,) = ax.plot([], [], color="tab:blue", lw=2, alpha=0.6, label="car path")
    goal_dot = ax.scatter([], [], s=180, c="gold", edgecolors="black", label="goal")
    car_dot = ax.scatter([], [], s=80, c="tab:red", edgecolors="black", label="car")
    (heading_line,) = ax.plot([], [], color="tab:red", lw=2)
    frame_text = ax.text(-2.15, 2.0, "", fontsize=10)
    ax.legend(loc="lower right")

    car_xs = []
    car_ys = []

    def init():
        path_line.set_data([], [])
        goal_dot.set_offsets(np.array([[0.0, 0.0]]))
        car_dot.set_offsets(np.array([[0.0, 0.0]]))
        heading_line.set_data([], [])
        frame_text.set_text("")
        return path_line, goal_dot, car_dot, heading_line, frame_text

    def update(i):
        s = sampled[i]
        x, y = s["car_xy"]
        gx, gy = s["goal_xy"]
        fx, fy = s["fwd_xy"]

        car_xs.append(x)
        car_ys.append(y)
        path_line.set_data(car_xs, car_ys)
        goal_dot.set_offsets(np.array([[gx, gy]]))
        car_dot.set_offsets(np.array([[x, y]]))
        heading_line.set_data([x, x + 0.22 * fx], [y, y + 0.22 * fy])
        frame_text.set_text(f"frame: {i + 1}/{len(sampled)}")
        return path_line, goal_dot, car_dot, heading_line, frame_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(sampled),
        init_func=init,
        interval=60,
        blit=True,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=18))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-policy", type=str, default="car_reinforce_policy.pt")
    parser.add_argument("--arbitrary-policy", type=str, default="car_arbitrary_goal_policy.pt")
    parser.add_argument("--fixed-out", type=str, default="media/fixed_goal_demo.gif")
    parser.add_argument("--arbitrary-out", type=str, default="media/arbitrary_goal_demo.gif")
    parser.add_argument("--frame-stride", type=int, default=6)
    parser.add_argument("--arbitrary-goal-x", type=float, default=1.2)
    parser.add_argument("--arbitrary-goal-y", type=float, default=0.8)
    args = parser.parse_args()

    fixed_traj = rollout_fixed(args.fixed_policy, max_steps=2200)
    save_trajectory_gif(
        fixed_traj,
        args.fixed_out,
        title="Fixed Goal Policy Demo",
        frame_stride=args.frame_stride,
    )

    goal = np.array([args.arbitrary_goal_x, args.arbitrary_goal_y], dtype=np.float32)
    arbitrary_traj = rollout_arbitrary(args.arbitrary_policy, goal_xy=goal, max_steps=2200)
    save_trajectory_gif(
        arbitrary_traj,
        args.arbitrary_out,
        title="Arbitrary Goal Policy Demo",
        frame_stride=args.frame_stride,
    )

    print(f"Saved fixed demo gif: {args.fixed_out}")
    print(f"Saved arbitrary demo gif: {args.arbitrary_out}")


if __name__ == "__main__":
    main()
