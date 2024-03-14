from abc import abstractmethod
import random
import sys
import numpy as np
from typing import Any, Dict, List, Optional
from core.typing import JointPolicy, Policy
from algorithms.base_nfg_algorithm import BaseNFGAlgorithm


class LyapunovBasedAlgorithm(BaseNFGAlgorithm):
    """A util class for all Lyapunov-regularization based algorithms"""

    @abstractmethod
    def get_eta(self, step : int) -> float:
        """This method should return the value of the eta parameter at the given step.
        It it necessary for any Lyapunov based algorithm to implement this method, as the value of eta is used to
        regularize the reward.

        Args:
            step (int): the current step

        Returns:
            float: the value of the eta parameter
        """

    def lyapunov_reward(
        self,
        chosen_actions: List[int],
        pi: JointPolicy,
        mu: JointPolicy,
    ) -> List[float]:
        """Implements the part of the Lyapunov reward modification that depends on the chosen actions.

        Args:
            player (int): the player who chose the action
            chosen_actions (List[int]): the chosen actions of the players
            pi (JointPolicy): the joint policy used to choose the actions
            mu (JointPolicy): the regularization joint policy
            eta (float): a parameter of the algorithm

        Returns:
            List[float]: the modified rewards
        """
        assert hasattr(self, "n_players"), "The number of players should be defined"
        lyap_reward = np.zeros(self.n_players)
        eta = self.get_eta()
        if eta == 0:
            return 0
        n_players = len(pi)
        eps = sys.float_info.epsilon
        for i in range(n_players):
            j = 1 - i
            act_i = chosen_actions[i]
            act_j = chosen_actions[j]
            lyap_reward[i] = (
                lyap_reward[i]
                - eta * np.log(pi[i][act_i] / mu[i][act_i] + eps)
                + eta * np.log(pi[j][act_j] / mu[j][act_j] + eps)
            )

        return lyap_reward

    def transform_q_value(self) -> None:
        assert hasattr(self, "joint_policy_mu"), "The joint policy mu should be defined"
        assert hasattr(self, "joint_policy_pi"), "The joint policy pi should be defined"
        assert hasattr(self, "joint_q_values"), "The joint q values should be defined"
        eta = self.get_eta()
        for player in range(len(self.joint_q_values)):
            opponent_policy = self.joint_policy_pi[1 - player]
            opponent_term = (
                opponent_policy
                * np.log(opponent_policy / self.joint_policy_mu[1 - player])
            ).sum()
            player_term = np.log(
                self.joint_policy_pi[player] / self.joint_policy_mu[player]
            )

            self.joint_q_values[player] = self.joint_q_values[player] + eta * (
                -player_term + opponent_term
            )
