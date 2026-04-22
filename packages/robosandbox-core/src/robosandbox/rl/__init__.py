"""RL training for robosandbox — PPO via Newton multi-world backend.

Requires PyTorch and the Newton sim backend.
"""

from robosandbox.rl.obs_encoder import ObsEncoder
from robosandbox.rl.ppo import ActorCritic, NeuralPolicy, train_ppo
from robosandbox.rl.reward import compute_shaped_reward

__all__ = [
    "ActorCritic",
    "NeuralPolicy",
    "ObsEncoder",
    "compute_shaped_reward",
    "train_ppo",
]
