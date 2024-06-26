import sys
import numpy as np
from typing import Any, Dict, List, Optional
from algorithms.algorithms_lyapunov import LyapunovBasedAlgorithm

from algorithms.forel import Forel
from core.nash import compute_nash_equilibrium
from core.online_plotter import DataPolicyToPlot
from core.scheduler import Linear, Scheduler
from core.utils import instantiate_class, to_numeric
from games.base_nfg_game import BaseNFGGame
from core.typing import JointPolicy
from copy import deepcopy


class IteratedForel(
    LyapunovBasedAlgorithm,
    Forel,
):
    def __init__(
        self,
        # FoReL specific parameters
        forel_config: Dict[str, Any],
        # Iterated FoReL specific parameters
        n_timesteps_per_iterations: int,
        do_mu_update: bool,
        do_linear_interpolation_mu: bool,
        eta_scheduler_config: dict,
        do_set_NE_as_init_mu: bool = False,
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
            do_set_NE_as_init_mu (bool): whether to set the Nash equilibrium as the mu policy at the end of each iteration, or not (for experimental purposes)
        """
        super().__init__(forel_config)
        self.n_timesteps_per_iterations = to_numeric(n_timesteps_per_iterations)
        self.do_mu_update = do_mu_update
        self.do_linear_interpolation_mu = do_linear_interpolation_mu
        self.eta_scheduler: Scheduler = instantiate_class(config=eta_scheduler_config)
        self.lyapunov = True
        self.do_set_NE_as_init_mu = do_set_NE_as_init_mu  # use Nash equilibrium as initial mu (for experimental purposes) =   do_set_NE_as_init_mu : False  # use Nash equilibrium as initial mu (for experimental purposes)

    # Interface methods

    def initialize_algorithm(
        self,
        game: BaseNFGGame,
        joint_policy_pi: Optional[JointPolicy] = None,
    ) -> None:
        super().initialize_algorithm(game=game, joint_policy_pi=joint_policy_pi)
        self.iteration: int = 0
        self.joint_policy_mu_k_minus_1 = self.initialize_randomly_joint_policy(
            n_actions=self.n_actions
        )
        if (
            self.do_set_NE_as_init_mu
        ):  # use Nash equilibrium as initial mu (for experimental purposes):
            self.joint_policy_mu = compute_nash_equilibrium(game, method="LP")
        else:
            self.joint_policy_mu = deepcopy(self.joint_policy_pi)

    def learn(
        self,
        joint_action: List[int],
        probs: List[float],
        rewards: List[float],
    ) -> None:

        # --- Modify the rewards ---

        # Compute the linear interpolation rate if needed
        if self.do_mu_update and self.do_linear_interpolation_mu:
            interpolation_rate_alpha = self.get_interpolation_rate_alpha()

        # Add the Lyapunov reward to the rewards
        lyap_reward_from_mu = self.lyapunov_reward(
            chosen_actions=joint_action,
            pi=self.joint_policy_pi,
            mu=self.joint_policy_mu,
        )

        # If we use linear interpolation, we also need to compute the Lyapunov reward from mu_{k-1}
        if self.do_mu_update and self.do_linear_interpolation_mu:
            lyap_reward_from_mu_k_minus_1 = self.lyapunov_reward(
                chosen_actions=joint_action,
                pi=self.joint_policy_pi,
                mu=self.joint_policy_mu_k_minus_1,
            )
            lyap_reward_from_mu = (
                (1 - interpolation_rate_alpha) * lyap_reward_from_mu_k_minus_1
                + interpolation_rate_alpha * lyap_reward_from_mu
            )

        # Add the Lyapunov reward to the rewards
        rewards += lyap_reward_from_mu

        # --- Do one learning step of FoReL ---
        metrics = super().learn(joint_action=joint_action, probs=probs, rewards=rewards)

        # --- At the end of the iteration, update mu and restart the FoReL algo (but keep the pi policy) ---
        if self.timestep == self.n_timesteps_per_iterations:
            if self.do_mu_update:
                self.joint_policy_mu_k_minus_1 = deepcopy(self.joint_policy_mu)
                self.joint_policy_mu = deepcopy(self.joint_policy_pi)
            self.iteration += 1
            super().initialize_algorithm(
                game=self.game,
                joint_policy_pi=self.joint_policy_pi,
            )

        # Add the metrics and dataPolicies to plot
        metrics["iteration"] = self.iteration
        metrics["timestep"] = self.timestep
        metrics["eta"] = self.get_eta()

        for i in range(self.n_players):
            metrics[f"reward_modif/reward_modif_{i}"] = rewards[i]
            for a in range(self.n_actions[i]):
                metrics[f"mu_{i}/mu_{i}_{a}"] = self.joint_policy_mu[i][a]
        metrics["mu"] = DataPolicyToPlot(
            name="μ",
            joint_policy=self.joint_policy_mu,
            color="g",
            marker="o",
            is_unique=True,
        )
        metrics["mu_k"] = DataPolicyToPlot(
            name="μ_k",
            joint_policy=self.joint_policy_mu_k_minus_1,
            color="g",
            marker="x",
        )
        if self.do_mu_update and self.do_linear_interpolation_mu:
            metrics["interpolation_rate_alpha"] = interpolation_rate_alpha
            metrics["mu_interp"] = DataPolicyToPlot(
                name="mu_interp",
                joint_policy=[
                    interpolation_rate_alpha * self.joint_policy_mu[i]
                    + (1 - interpolation_rate_alpha) * self.joint_policy_mu_k_minus_1[i]
                    for i in range(self.n_players)
                ],
                color="c",
                marker="o",
                is_unique=True,
            )
        return metrics

    # Helper methods
    def get_eta(self) -> float:
        return self.eta_scheduler.get_value(
            self.iteration * self.n_timesteps_per_iterations + self.timestep
        )

    def get_interpolation_rate_alpha(self) -> float:
        return Linear(
            start_value=0,
            end_value=1,
            n_steps=self.n_timesteps_per_iterations // 2,
            upper_bound=1,
            lower_bound=0,
        ).get_value(self.timestep)
