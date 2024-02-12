from typing import List

import numpy as np
from games.base_nfg_game import BaseNFGGame


class MatchingPennies(BaseNFGGame):

    def __init__(self, payoff_matrix: List[List[float]]) -> None:
        self.payoff_matrix = payoff_matrix
        super().__init__()

    def get_rewards(self, joint_action: List[int]) -> List[float]:
        return self.payoff_matrix[joint_action[0]][joint_action[1]]

    def num_distinct_actions(self) -> int:
        return 2

    def num_players(self) -> int:
        return 2

    def get_utility_matrix(self) -> np.ndarray:
        return np.array(self.payoff_matrix, dtype=float)
