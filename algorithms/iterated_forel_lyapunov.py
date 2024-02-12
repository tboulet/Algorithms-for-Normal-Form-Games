import sys
from time import sleep
from matplotlib import pyplot as plt
import numpy as np
from typing import Any, Dict, List, Callable, Tuple

from algorithms.base_nfg_algorithm import BaseNFGAlgorithm
from algorithms.forel import Forel
from core.online_plotter import PointToPlot
from core.scheduler import Scheduler
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
        do_mu_update: bool,
        do_linear_interpolation_mu: bool,
        eta_scheduler_config: dict,
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
            do_mu_update (bool): whether to update the mu policy at each iteration, or not (in this case the mu used is the initial pi policy)
            do_linear_interpolation_mu (bool): whether to use a linear interpolation between mu_{k-1} and mu_k, or not (in this case mu is brutally updated at the end of each iteration)
            eta_scheduler_config (dict): the configuration of the eta scheduler. It should contain the following keys:
                - type (str): the type of scheduler (either "constant" or "linear")
                - start_value (float): the initial value of eta
                - end_value (float): the final value of eta
                - n_iterations (int): the number of iterations over which eta will decrease
        """
        super().__init__(**forel_config)
        self.n_timesteps_per_iterations = to_numeric(n_timesteps_per_iterations)
        self.do_mu_update = do_mu_update
        self.do_linear_interpolation_mu = do_linear_interpolation_mu
        self.eta_scheduler = Scheduler(**eta_scheduler_config)

    # Interface methods

    def initialize_algorithm(
        self,
        game: BaseNFGGame,
    ) -> None:
        super().initialize_algorithm(game=game)
        self.iteration: int = 0
        self.joint_policy_mu_k_minus_1 = self.initialize_randomly_joint_policy(
            n_players=self.n_players, n_actions=self.n_actions
        )
        self.joint_policy_mu = self.joint_policy_pi.copy()

    def learn(
        self,
        joint_action: List[int],
        probs: List[float],
        rewards: List[float],
    ) -> None:

        # --- Modify the rewards ---
        if self.do_mu_update:

            # Compute the linear interpolation rate if needed
            if self.do_linear_interpolation_mu:
                alpha = Scheduler(
                    type="linear",
                    start_value=0,
                    end_value=self.n_timesteps_per_iterations // 2,
                    n_steps=self.n_timesteps_per_iterations / 2,
                ).get_value(
                    self.timestep - self.iteration * self.n_timesteps_per_iterations
                )

            # Add the Lyapunov reward to the rewards
            for i in range(self.n_players):
                lyap_reward_from_mu = self.lyapunov_reward(
                    player=i,
                    chosen_actions=joint_action,
                    pi=self.joint_policy_pi,
                    mu=self.joint_policy_mu,
                    eta=self.eta_scheduler.get_value(self.timestep),
                )

                # If we use linear interpolation, we also need to compute the Lyapunov reward from mu_{k-1}
                if self.do_linear_interpolation_mu:
                    lyap_reward_from_mu_k_minus_1 = self.lyapunov_reward(
                        player=i,
                        chosen_actions=joint_action,
                        pi=self.joint_policy_pi,
                        mu=self.joint_policy_mu_k_minus_1,
                        eta=self.eta_scheduler.get_value(self.timestep),
                    )
                    lyap_reward_from_mu = (
                        1 - alpha
                    ) * lyap_reward_from_mu_k_minus_1 + alpha * lyap_reward_from_mu

                # Add the Lyapunov reward to the rewards
                rewards[i] += lyap_reward_from_mu

        # --- Do one learning step of FoReL ---
        metrics = super().learn(joint_action=joint_action, probs=probs, rewards=rewards)

        # --- At the end of the iteration, update mu and restart the FoReL algo (but keep the pi policy) ---
        if self.timestep == self.n_timesteps_per_iterations:
            self.joint_policy_mu_k_minus_1 = self.joint_policy_mu.copy()
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
        metrics["mu_k"] = PointToPlot(
            name="μ_k",
            coords=self.joint_policy_mu[:, 0],
            color="g",
            marker="o",
            is_unique=True,
        )
        metrics["mu_{k-1}"] = PointToPlot(
            name="μ_{k-1}",
            coords=self.joint_policy_mu_k_minus_1[:, 0],
            color="g",
            marker="x",
            is_unique=True,
        )
        return metrics

    # Helper methods

    def lyapunov_reward(
        self,
        player: int,
        chosen_actions: List[int],
        pi: JointPolicy,
        mu: JointPolicy,
        eta: float,
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

        if eta == 0:
            return 0
        n_players = len(pi)
        eps = sys.float_info.epsilon
        # Compute the probs for the player and the others, for pi and mu
        pi_i_a = pi[player][chosen_actions[player]]
        pi_minus_i_a = np.prod(
            [pi[j][chosen_actions[j]] for j in range(n_players) if j != player]
        )
        mu_i_a = mu[player][chosen_actions[player]]
        mu_minus_i_a = np.prod(
            [mu[j][chosen_actions[j]] for j in range(n_players) if j != player]
        )
        # Compute the Lyapunov reward modification
        return -eta * np.log(pi_i_a / mu_i_a + eps) + eta * np.log(
            pi_minus_i_a / mu_minus_i_a + eps
        )

    def get_model_based_q_value(
        self, game: BaseNFGGame, player: int, action: int, joint_policy: JointPolicy
    ) -> float:
        assert self.n_players == 2, "This algorithm only works for 2-player games"
        # Compute the Q value using the model-based method
        q_value = super().get_model_based_q_value(game, player, action, joint_policy)
        # Add the expected Lyapunov reward
        for b in range(game.num_distinct_actions()):
            q_value += joint_policy[1 - player][b] * self.lyapunov_reward(
                player=player,
                chosen_actions=[action, b] if player == 0 else [b, action],
                pi=joint_policy,
                mu=self.joint_policy_mu,
                eta=self.eta_scheduler.get_value(self.timestep),
            )
        return q_value
