import sys
from time import sleep
from matplotlib import pyplot as plt
import numpy as np
from typing import Any, Dict, List, Callable, Tuple

from algorithms.base_nfg_algorithm import BaseNFGAlgorithm
from algorithms.forel import Forel
from core.online_plotter import PointToPlot
from games.base_nfg_game import BaseNFGGame
from core.typing import JointPolicy, Policy
from core.utils import to_numeric


class IteratedForel(Forel):
    def __init__(
        self,
        # FoReL specific parameters
        forel_config: Dict[str, Any],
        # Iterated FoReL specific parameters
        n_timesteps_per_iterations: int,
        eta: float,
    ) -> None:
        """Initializes the Iterated FoReL algorithm.

        Args:
            forel_config (Dict[str, Any]): the configuration of the FoReL algorithm. It should contain the following keys:
                - q_value_estimation_method (str): the method used to estimate the Q values (either "mc" or "model-based")
                - dynamics_method (str): the method used to update the policy (either "softmax" or "rd" (Replicator Dynamics))
                - learning_rate_rd (float): the learning rate used to update the policy (only used if dynamics_method == "rd")
                - learning_rate_cum_values (float): the learning rate used to update the cumulative values (only used if dynamics_method == "softmax")
                - n_monte_carlo_q_evaluation (int): the number of episodes used to estimate the Q values
                - regularizer (str): the regularizer function tag (for now either "entropy" or "l2")
            n_timesteps_per_iterations (int): the number of timesteps per iteration
            eta (float): the eta parameter of the algorithm for modifying the rewards
        """
        super().__init__(**forel_config)
        self.n_timesteps_per_iterations = n_timesteps_per_iterations
        self.eta = eta
        self.lyapunov = True

    # Interface methods

    def initialize_algorithm(
        self,
        game: BaseNFGGame,
    ) -> None:
        super().initialize_algorithm(game=game)
        self.iteration: int = 0
        self.joint_policy_mu = self.initialize_randomly_joint_policy(
            n_actions=self.n_actions
        )

    def learn(
        self,
        joint_action: List[int],
        probs: List[float],
        rewards: List[float],
    ) -> None:

        # --- Modify the rewards ---
        rewards = self.modify_rewards(
            returns=rewards,
            chosen_actions=joint_action,
            pi=self.joint_policy_pi,
            mu=self.joint_policy_mu,
            eta=self.eta,
        )

        # --- Do one learning step of FoReL ---
        metrics = super().learn(joint_action=joint_action, probs=probs, rewards=rewards)

        # --- At the end of the iteration, update mu and restart the FoReL algo (but keep the pi policy) ---
        if self.timestep == self.n_timesteps_per_iterations:
            self.joint_policy_mu = self.joint_policy_pi.copy()
            self.iteration += 1
            super().initialize_algorithm(
                game=self.game,
                joint_policy_pi=self.joint_policy_pi,
            )

        # Add the mu probs to the metrics as well as the current point
        metrics.update(
            {f"mu_0(a={a})": self.joint_policy_mu[0][a] for a in range(self.n_actions)}
        )
        metrics["mu_point"] = PointToPlot(
            name="mu",
            coords=self.joint_policy_mu[:, 0],
            color="g",
            marker="o",
            is_unique=True,
        )
        return metrics

    # Helper methods

    def modify_rewards(
        self,
        returns: List[float],
        chosen_actions: List[int],
        pi: JointPolicy,
        mu: JointPolicy,
        eta: float,
    ) -> List[float]:
        """Implements the modification of rewards for the Forel algorithm.

        Args:
            returns (List[float]): the rewards obtained by the players
            chosen_actions (List[int]): the actions chosen by the players
            pi (JointPolicy): the joint policy used to choose the actions
            mu (JointPolicy): the regularization joint policy
            eta (float): a parameter of the algorithm

        Returns:
            List[float]: the modified rewards
        """
        returns_modified = returns.copy()

        if eta == 0:
            return returns_modified

        n_players = len(pi)

        eps = sys.float_info.epsilon
        for i in range(n_players):
            pi_i_a = pi[i][chosen_actions[i]]
            pi_minus_i_a = np.prod(
                [pi[j][chosen_actions[j]] for j in range(n_players) if j != i]
            )
            mu_i_a = mu[i][chosen_actions[i]]
            mu_minus_i_a = np.prod(
                [mu[j][chosen_actions[j]] for j in range(n_players) if j != i]
            )
            returns_modified[i] = (
                returns_modified[i]
                - eta * np.log(pi_i_a / mu_i_a + eps)
                + eta * np.log(pi_minus_i_a / mu_minus_i_a + eps)
            )

        return returns_modified

    def transform_q_value(self) -> None:
        for player in range(len(self.joint_q_values)):
            opponent_policy = self.joint_policy_pi[1 - player]
            oppenent_term = (
                opponent_policy
                * np.log(opponent_policy / self.joint_policy_mu[1 - player]).sum()
            )
            player_term = np.log(
                self.joint_policy_pi[player] / self.joint_policy_mu[player]
            )

            self.joint_q_values[player] = self.joint_q_values[player] + self.eta * (
                -player_term + oppenent_term
            )
