import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn


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
        self.np_random = np.random.default_rng(0)
        self.set_goal(self.goal_xy)

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

    def get_obs(self):
        car_xy = self.data.body("car").xpos[:2]
        qvel = self.data.qvel
        forward_xy = self.data.body("car").xmat.reshape(3, 3)[:, 0][:2]
        goal_delta = self.goal_xy - car_xy
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

    def distance_to_goal(self):
        car_xy = self.data.body("car").xpos[:2]
        return float(np.linalg.norm(car_xy - self.goal_xy))

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
        return self.get_obs(), {}

    def step(self, action):
        self.step_count += 1
        self.data.ctrl[:] = np.tanh(np.asarray(action, dtype=np.float32))
        mujoco.mj_step(self.model, self.data)
        dist = self.distance_to_goal()
        terminated = dist < self.goal_radius
        truncated = self.step_count >= self.max_episode_steps
        info = {
            "distance_to_goal": dist,
            "success": float(terminated),
        }
        return self.get_obs(), 0.0, terminated, truncated, info


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu")
    policy = ActorCritic(obs_dim=checkpoint["obs_dim"], act_dim=checkpoint["action_dim"])
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return policy, checkpoint


def main(policy_path, fixed_goal=None):
    env = ArbitraryGoalCars(max_episode_steps=2200)
    policy, meta = load_policy(policy_path)
    obs, _ = env.reset(seed=0, goal_xy=fixed_goal)

    print(
        "Viewer opening... Press ESC to close. "
        f"(policy success_rate={meta.get('success_rate', 0.0):.2f})"
    )

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                residual = policy.policy_mean(policy.policy_backbone(obs_t))[0].numpy()
                action = heuristic_action(obs) + 0.2 * residual

            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                if fixed_goal is None:
                    obs, _ = env.reset(goal_xy=None)
                else:
                    obs, _ = env.reset(goal_xy=np.asarray(fixed_goal, dtype=np.float32))
                print(
                    f"reset | success={bool(info['success'])} "
                    f"final_dist={info['distance_to_goal']:.3f} "
                    f"goal=({env.goal_xy[0]:.2f},{env.goal_xy[1]:.2f})"
                )

            viewer.sync()
            dt = env.model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="car_arbitrary_goal_policy.pt")
    parser.add_argument("--goal-x", type=float, default=None)
    parser.add_argument("--goal-y", type=float, default=None)
    args = parser.parse_args()

    fixed_goal = None
    if args.goal_x is not None and args.goal_y is not None:
        fixed_goal = np.array([args.goal_x, args.goal_y], dtype=np.float32)

    main(policy_path=args.policy, fixed_goal=fixed_goal)
