import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

from train_arbitrary_goals import ActorCritic, ArbitraryGoalCars, heuristic_action


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
