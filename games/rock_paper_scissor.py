from typing import List

import numpy as np
from games.base_nfg_game import BaseNFGGame


class RockPaperScissor(BaseNFGGame):

    def __init__(self, payoff_matrix: List[List[float]]) -> None:
        self.payoff_matrix = np.array(payoff_matrix, dtype=np.float64)
        assert self.payoff_matrix.shape == (3, 3, 2), f"Payoff matrix has shape {self.payoff_matrix.shape} but it should be (3, 3, 2)"
        super().__init__()

    def get_rewards(self, joint_action: List[int]) -> np.ndarray:
        return self.payoff_matrix[tuple(joint_action)].copy()

    def num_distinct_actions(self) -> List[int]:
        return [3, 3]

    def num_players(self) -> int:
        return 2

    def get_utility_matrix(self) -> np.ndarray:
        return self.payoff_matrix.copy()
