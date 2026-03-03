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
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.8))
        self.value_head = nn.Linear(128, 1)

    def deterministic_action(self, x):
        features = self.policy_backbone(x)
        return self.policy_mean(features)


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


class Cars:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("./car.xml")
        self.data = mujoco.MjData(self.model)
        self.goal_xy = np.array([1.0, -1.0], dtype=np.float32)
        self.goal_radius = 0.16
        mujoco.mj_forward(self.model, self.data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = 0.03
        self.data.qpos[3] = 1.0
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

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

    def step(self, action):
        self.data.ctrl[:] = np.tanh(np.asarray(action, dtype=np.float32))
        mujoco.mj_step(self.model, self.data)
        return self.get_obs()

    def distance_to_goal(self):
        car_xy = self.data.body("car").xpos[:2]
        return float(np.linalg.norm(car_xy - self.goal_xy))


def load_policy(path="car_reinforce_policy.pt"):
    checkpoint = torch.load(path, map_location="cpu")
    policy = ActorCritic(obs_dim=checkpoint["obs_dim"], act_dim=checkpoint["action_dim"])
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return policy, checkpoint


def main():
    env = Cars()
    policy, metadata = load_policy("car_reinforce_policy.pt")
    state = env.reset()

    print(
        "Viewer opening... Press ESC to close. "
        f"(trained success_rate={metadata.get('success_rate', 0.0):.2f})"
    )

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                residual = policy.deterministic_action(state_t)[0].numpy()
                action = heuristic_action(state) + 0.2 * residual

            state = env.step(action)
            dist = env.distance_to_goal()
            if dist < env.goal_radius:
                print(f"Reached goal (distance={dist:.3f}). Resetting.")
                state = env.reset()

            viewer.sync()
            dt = env.model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"Viewer failed to launch: {exc}")
        env = Cars()
        policy, _ = load_policy("car_reinforce_policy.pt")
        env.reset()

        def controller(model, data):
            car_xy = data.body("car").xpos[:2]
            qvel = data.qvel
            forward_xy = data.body("car").xmat.reshape(3, 3)[:, 0][:2]
            goal_delta = env.goal_xy - car_xy
            state = np.array(
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
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                residual = policy.deterministic_action(state_t)[0].numpy()
                action = heuristic_action(state) + 0.2 * residual
            data.ctrl[:] = np.tanh(action)

        mujoco.set_mjcb_control(controller)
        mujoco.viewer.launch(env.model, env.data)
