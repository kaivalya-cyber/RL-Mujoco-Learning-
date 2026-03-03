import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")


class SmokeTests(unittest.TestCase):
    def test_imports(self):
        import evaluate  # noqa: F401
        import generate_demo_gifs  # noqa: F401
        import generate_report  # noqa: F401
        import main  # noqa: F401
        import main_arbitrary_goals  # noqa: F401
        import train  # noqa: F401
        import train_arbitrary_goals  # noqa: F401

    def test_fixed_env_reset_step(self):
        from train import Cars

        env = Cars(max_episode_steps=10, random_start=True)
        obs, _ = env.reset(seed=7)
        self.assertEqual(obs.shape, (9,))

        next_obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        self.assertEqual(next_obs.shape, (9,))
        self.assertIsInstance(float(reward), float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("distance_to_goal", info)

    def test_arbitrary_env_reset_step(self):
        from train_arbitrary_goals import ArbitraryGoalCars

        env = ArbitraryGoalCars(max_episode_steps=10)
        obs, _ = env.reset(seed=11)
        self.assertEqual(obs.shape, (9,))

        next_obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        self.assertEqual(next_obs.shape, (9,))
        self.assertIsInstance(float(reward), float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("goal_x", info)
        self.assertIn("goal_y", info)

    def test_checkpoint_load_paths(self):
        import main
        import main_arbitrary_goals

        fixed = Path("car_reinforce_policy.pt")
        arbitrary = Path("car_arbitrary_goal_policy.pt")
        self.assertTrue(fixed.exists(), "Expected fixed checkpoint to exist")
        self.assertTrue(arbitrary.exists(), "Expected arbitrary checkpoint to exist")

        policy1, meta1 = main.load_policy(str(fixed))
        self.assertIsNotNone(policy1)
        self.assertIn("obs_dim", meta1)

        policy2, meta2 = main_arbitrary_goals.load_policy(str(arbitrary))
        self.assertIsNotNone(policy2)
        self.assertIn("obs_dim", meta2)

    def test_short_train_steps(self):
        import train
        import train_arbitrary_goals

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            fixed_ckpt = td_path / "fixed.pt"
            fixed_curve = td_path / "fixed.png"
            train.train(
                total_episodes=1,
                max_episode_steps=20,
                eval_every=100,
                checkpoint_path=str(fixed_ckpt),
                curve_path=str(fixed_curve),
            )
            self.assertTrue(fixed_ckpt.exists())
            self.assertTrue(fixed_curve.exists())

            arb_ckpt = td_path / "arb.pt"
            arb_curve = td_path / "arb.png"
            arb_summary = td_path / "arb.json"
            train_arbitrary_goals.train(
                total_episodes=1,
                max_episode_steps=20,
                eval_every=100,
                checkpoint_path=str(arb_ckpt),
                curve_path=str(arb_curve),
                summary_path=str(arb_summary),
            )
            self.assertTrue(arb_ckpt.exists())
            self.assertTrue(arb_curve.exists())
            self.assertTrue(arb_summary.exists())


if __name__ == "__main__":
    unittest.main()
