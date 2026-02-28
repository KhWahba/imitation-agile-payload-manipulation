import os
import tempfile
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.data.wrappers import RolloutInfoWrapper

from point2d_env import Point2DEnv
from expert_pd_controller import PDExpert, ExpertPolicySB3
import torch as th
from pathlib import Path

def main():
    rng = np.random.default_rng(0)
    xml_path = "../envs/point2d.xml"

    # 1) VecEnv
    def make_env():
        return RolloutInfoWrapper(
            Point2DEnv(
                xml_path=xml_path,
                goal=(0.8, -0.2),
                max_steps=300,
            )
        )

    venv = DummyVecEnv([make_env])  # n_envs=1

    # 2) Expert (SB3 policy wrapper)
    expert = PDExpert(
        kp=6.0,
        kd=3.0,
        act_low=venv.action_space.low,
        act_high=venv.action_space.high,
    )
    expert_policy = ExpertPolicySB3(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        expert=expert,
        device="cpu",
    )

    # 3) BC trainer (learner)
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        rng=rng,
    )

    # 4) DAgger trainer
    scratch_root = "runs/dagger_point2d"
    os.makedirs(scratch_root, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=scratch_root, prefix="scratch_",) as tmpdir:
        print(tmpdir)
        dagger_trainer = SimpleDAggerTrainer(
            venv=venv,
            scratch_dir=tmpdir,
            expert_policy=expert_policy,
            bc_trainer=bc_trainer,
            rng=rng,
        )

        dagger_trainer.train(8000)  # train for 8000 steps

        # quick eval (SB3 utility expects VecEnv)
        reward, _ = evaluate_policy(dagger_trainer.policy, venv, n_eval_episodes=10)
        print("Eval reward:", reward)

        # save learner policy (imitation policies are SB3-style)
        save_path = Path("runs/dagger_point2d")
        model_name = "model.zip"
        th.save(dagger_trainer.policy.state_dict(), "runs/dagger_point2d/policy_state_dict.pt")
        print("Saved learner policy to:", save_path)


if __name__ == "__main__":
    main()
