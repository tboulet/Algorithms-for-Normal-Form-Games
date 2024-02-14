from abc import ABC, abstractmethod
from typing import List
import numpy as np

from core.typing import JointPolicy, Policy


class BaseNFGGame(ABC):
    """The base class for any Normal Form Game"""

    @abstractmethod
    def get_rewards(self, joint_action: List[int]) -> List[float]:
        """Return the rewards for each player"""
        pass

    @abstractmethod
    def num_distinct_actions(self) -> List[int]:
        """Return the number of distinct actions for each player"""
        pass

    @abstractmethod
    def num_players(self) -> int:
        """Return the number of players"""
        pass

    def get_utility_matrix(self) -> np.ndarray:
        """Return the utility matrix"""
        raise NotImplementedError("This game does not have a utility matrix")

    def get_model_based_q_value(
        self, player: int, joint_policy: JointPolicy
    ) -> np.ndarray:
        """Return the Q-value of a player's action given a joint policy"""

        utility_matrix = self.get_utility_matrix()

        if player == 0:
            weighted_utility = utility_matrix[:, :, 0] * joint_policy[1][:, np.newaxis]
            return np.sum(weighted_utility, axis=0)

        weighted_utility = utility_matrix[:, :, 1] * joint_policy[0][np.newaxis, :]
        return np.sum(weighted_utility, axis=1)
