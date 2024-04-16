import random
import sys
import numpy as np
from typing import Any, Dict, List, Optional
from algorithms.algorithms_lyapunov import LyapunovBasedAlgorithm
from algorithms.algorithms_population import PopulationBasedAlgorithm

from algorithms.forel import Forel
from core.online_plotter import DataPolicyToPlot
from core.scheduler import Scheduler
from core.utils import instantiate_class, to_numeric, try_get
from games.base_nfg_game import BaseNFGGame
from core.typing import JointPolicy, Policy
from copy import deepcopy


class PopulationIteratedLyapunovForel(
    LyapunovBasedAlgorithm,
    PopulationBasedAlgorithm,
    Forel,
):
    def __init__(
        self,
        # FoReL specific parameters
        forel_config: Dict[str, Any],
        # Population parameters
        population_averaging: str,
        n_last_policies_to_sample : int,
        sampler_population: Dict[str, Any],
        # PIL-FoReL specific parameters
        n_timesteps_per_iterations: int,
        eta_scheduler_config: dict,
    ) -> None:
        """Initializes the Population Iterated Lyapunov FoReL algorithm.

        Args:
            forel_config (Dict[str, Any]): the configuration of the FoReL algorithm. It should contain the following keys:
                - q_value_estimation_method (str): the method used to estimate the Q values (either "mc" or "model-based")
                - dynamics_method (str): the method used to update the policy (either "softmax" or "rd" (Replicator Dynamics))
                - learning_rate_rd (float): the learning rate used to update the policy (only used if dynamics_method == "rd")
                - learning_rate_cum_values (float): the learning rate used to update the cumulative values (only used if dynamics_method == "softmax")
                - n_monte_carlo_q_evaluation (int): the number of episodes used to estimate the Q values
                - regularizer (str): the regularizer function tag (for now either "entropy" or "l2")
            n_timesteps_per_iterations (int): the number of timesteps per iteration
            population_averaging (str): the type of averaging used to update the population (either "geometric" or "arithmetic")
            n_last_policies_to_sample (int): the number of last policies to sample
            sampler_population (Dict[str, Any]): the configuration of the population sampler.
            eta_scheduler_config (dict): the configuration of the eta scheduler.
        """
        Forel.__init__(self, forel_config)
        PopulationBasedAlgorithm.__init__(
            self,
            sampler_population=sampler_population,
            population_averaging=population_averaging,
            n_last_policies_to_sample=n_last_policies_to_sample,
        )
        LyapunovBasedAlgorithm.__init__(self)

        self.n_timesteps_per_iterations = to_numeric(n_timesteps_per_iterations)
        self.eta_scheduler: Scheduler = instantiate_class(config=eta_scheduler_config)
        self.lyapunov = True

    def initialize_algorithm(
        self,
        game: BaseNFGGame,
        joint_policy_pi: Optional[JointPolicy] = None,
    ) -> None:
        self.metrics = {}
        super().initialize_algorithm(game=game, joint_policy_pi=joint_policy_pi)
        self.iteration: int = 0
        self.joint_policy_mu = deepcopy(self.joint_policy_pi)
        self.population = [deepcopy(self.joint_policy_pi)]
        self.kept_policies = []

    def learn(
        self,
        joint_action: List[int],
        probs: List[float],
        rewards: List[float],
    ) -> None:

        # --- Modify the rewards ---
        rewards += self.lyapunov_reward(
            chosen_actions=joint_action,
            pi=self.joint_policy_pi,
            mu=self.joint_policy_mu,
        )

        # --- Do one learning step of FoReL ---
        metrics_forel = super().learn(
            joint_action=joint_action, probs=probs, rewards=rewards
        )
        self.metrics.update(metrics_forel)

        # --- Update the population ---
        self.population.append(deepcopy(self.joint_policy_pi))

        # --- At the end of the iteration, update mu and restart the FoReL algo (but keep the pi policy) ---
        if self.timestep == self.n_timesteps_per_iterations:
            self.kept_policies = self.sample_policies()
            self.joint_policy_mu = self.average_list_of_joint_policies(
                self.kept_policies
            )
            self.iteration += 1
            super().initialize_algorithm(
                game=self.game,
                joint_policy_pi=self.joint_policy_pi,
            )

            # Add dataPolicies to plot
            self.metrics["pi_sampled"] = DataPolicyToPlot(
                name="π_sampled",
                joint_policy=self.kept_policies,
                color="black",
                marker="o",
                is_unique=True,
            )

        # Add the metrics and dataPolicies to plot
        self.metrics["iteration"] = self.iteration
        self.metrics["timestep"] = self.timestep
        self.metrics["eta"] = self.get_eta()

        for i in range(self.n_players):
            self.metrics[f"reward_modif/reward_modif_{i}"] = rewards[i]
            for a in range(self.n_actions[i]):
                self.metrics[f"mu_{i}/mu_{i}_{a}"] = self.joint_policy_mu[i][a]
        self.metrics["mu"] = DataPolicyToPlot(
            name="μ",
            joint_policy=self.joint_policy_mu,
            color="g",
            marker="o",
            is_unique=True,
        )
        self.metrics["mu_k"] = DataPolicyToPlot(
            name="μ_k",
            joint_policy=self.joint_policy_mu,
            color="g",
            marker="x",
        )
        return self.metrics

    # Helper methods
    def get_eta(self) -> float:
        return self.eta_scheduler.get_value(
            self.iteration * self.n_timesteps_per_iterations + self.timestep
        )
