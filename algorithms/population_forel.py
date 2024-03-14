from ast import For
import sys
import numpy as np
from typing import Any, Dict, List, Optional
import random
from algorithms.algorithms_population import PopulationBasedAlgorithm


from algorithms.forel import Forel
from core.online_plotter import DataPolicyToPlot
from core.scheduler import Scheduler
from core.utils import to_numeric, try_get
from games.base_nfg_game import BaseNFGGame
from core.typing import JointPolicy, Policy
from copy import deepcopy


class PopulationForel(
    PopulationBasedAlgorithm,
    Forel,
):
    def __init__(
        self,
        # FoReL specific parameters
        forel_config: Dict[str, Any],
        # Population parameters
        population_averaging: str,
        sampler_population: Dict[str, Any],
        # P-FoReL specific parameters
        population_timesteps_per_iterations: int,
        do_population_update: bool,
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
            population_timesteps_per_iterations (int): the number of timesteps per iteration
            do_population_update (bool): whether to update the population at each iteration, or not
            sampler_population (Dict[str, Any]): the configuration of the population sampler.
            population_averaging (str): the type of averaging used to update the population (either "geometric" or "arithmetic")
        """
        Forel.__init__(self, **forel_config)
        PopulationBasedAlgorithm.__init__(
            self,
            sampler_population=sampler_population,
            population_averaging=population_averaging,
        )

        self.population_timesteps_per_iterations = population_timesteps_per_iterations
        self.do_population_update = do_population_update
        self.lyapunov = False

    def initialize_algorithm(
        self,
        game: BaseNFGGame,
        joint_policy_pi: Optional[JointPolicy] = None,
    ) -> None:
        self.metrics = {}
        super().initialize_algorithm(game=game, joint_policy_pi=joint_policy_pi)
        self.iteration: int = 0
        self.population = [deepcopy(self.joint_policy_pi)]
        self.kept_policies = []

    def learn(
        self,
        joint_action: List[int],
        probs: List[float],
        rewards: List[float],
    ) -> None:

        # --- Do one learning step of FoReL ---
        metrics_forel = super().learn(
            joint_action=joint_action, probs=probs, rewards=rewards
        )
        self.metrics.update(metrics_forel)

        self.population.append(deepcopy(self.joint_policy_pi))

        # --- At the end of the iteration, update pi policy and restart the FoReL algo (but keep the pi policy) ---
        if (
            self.do_population_update
            and self.timestep == self.population_timesteps_per_iterations
        ):
            self.kept_policies = self.sample_policies()
            self.joint_policy_pi = self.average_list_of_joint_policies(
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

        return self.metrics
