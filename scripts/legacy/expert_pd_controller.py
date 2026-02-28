import numpy as np
import torch as th
from stable_baselines3.common.policies import BasePolicy


class PDExpert:
    """u = -Kp*(p-g) - Kd*v"""
    def __init__(self, kp=6.0, kd=3.0, act_low=None, act_high=None):
        self.kp = float(kp)
        self.kd = float(kd)
        self.act_low = None if act_low is None else np.asarray(act_low, dtype=np.float32)
        self.act_high = None if act_high is None else np.asarray(act_high, dtype=np.float32)

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs.shape[0] < 6:
            raise ValueError(f"Expected obs dim >= 6, got {obs.shape[0]}")
        x, y, xdot, ydot, gx, gy = obs[:6]
        p_err = np.array([x - gx, y - gy], dtype=np.float32)
        v = np.array([xdot, ydot], dtype=np.float32)
        u = -self.kp * p_err - self.kd * v
        if self.act_low is not None:
            u = np.maximum(u, self.act_low)
        if self.act_high is not None:
            u = np.minimum(u, self.act_high)
        return u.astype(np.float32)


class ExpertPolicySB3(BasePolicy):
    """Wrap PDExpert as an SB3 policy so imitation.dagger can call it."""
    def __init__(self, observation_space, action_space, expert: PDExpert, device="cpu"):
        super().__init__(observation_space, action_space)
        self.expert = expert
        self._device = th.device(device)  

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        """
        observation: torch tensor of shape (n_envs, obs_dim) OR (obs_dim,)
        returns: torch tensor of shape (n_envs, act_dim) OR (act_dim,)
        """
        # Ensure batch dimension
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)

        obs_np = observation.detach().cpu().numpy()
        acts = [self.expert.act(o) for o in obs_np]
        acts_np = np.stack(acts, axis=0).astype(np.float32)

        return th.as_tensor(acts_np, dtype=th.float32, device=observation.device)
