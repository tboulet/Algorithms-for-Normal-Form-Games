import sys
import numpy as np
from typing import Any, Dict, List, Optional
import random


from algorithms.forel import Forel
from core.online_plotter import DataPolicyToPlot
from core.scheduler import Scheduler
from core.utils import to_numeric
from games.base_nfg_game import BaseNFGGame
from core.typing import JointPolicy
from copy import deepcopy


class PopulationForel(Forel):
    def __init__(
        self,
        forel_config: Dict[str, Any],
        # Population FoReL specific parameters
        population_timesteps_per_iterations: int,
        do_population_update: bool,
        population_averaging: str = "geometric",
        population_k: int = 5,
        population_sampling: str = "random",
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
            population_averaging (str): the type of averaging used to update the population (either "geometric" or "arithmetic")
            population_k (int): the number of policies in the population
            population_sampling (str): the type of sampling used to update the population (either "random" or "greedy")
        """
        super().__init__(**forel_config)
        self.population_timesteps_per_iterations = to_numeric(
            population_timesteps_per_iterations
        )
        self.do_population_update = do_population_update
        self.population_averaging = population_averaging
        self.population_k = population_k
        self.population_sampling = population_sampling

        self.lyapunov = False

    def initialize_algorithm(
        self,
        game: BaseNFGGame,
        joint_policy_pi: Optional[JointPolicy] = None,
    ) -> None:
        super().initialize_algorithm(game=game, joint_policy_pi=joint_policy_pi)
        self.iteration: int = 0
        self.population = [deepcopy(self.joint_policy_pi)]

    def learn(
        self,
        joint_action: List[int],
        probs: List[float],
        rewards: List[float],
    ) -> None:

        # --- Do one learning step of FoReL ---
        metrics = super().learn(joint_action=joint_action, probs=probs, rewards=rewards)

        self.population.append(deepcopy(self.joint_policy_pi))

        # --- At the end of the iteration, update pi policy and restart the FoReL algo (but keep the pi policy) ---
        if self.timestep == self.population_timesteps_per_iterations:
            kept_policies = self.sample_policies()
            self.joint_policy_pi = self.average_policies(kept_policies)

            self.iteration += 1
            super().initialize_algorithm(
                game=self.game,
                joint_policy_pi=self.joint_policy_pi,
            )

        # Add the metrics and dataPolicies to plot
        metrics["iteration"] = self.iteration
        metrics["timestep"] = self.timestep

        return metrics

    def sample_policies(self) -> List[JointPolicy]:
        if self.population_sampling == "random":
            return random.sample(self.population, self.population_k)

        elif self.population_sampling == "greedy":
            raise NotImplementedError("Greedy sampling not implemented yet")

        else:
            raise ValueError(
                f"Unknown population_sampling method {self.population_sampling}"
            )

    def average_policies(self, joint_policies: List[JointPolicy]) -> JointPolicy:
        if self.population_averaging == "geometric":
            averaged_policy = [
                np.ones_like(player_policy) for player_policy in joint_policies[0]
            ]

            for i in range(len(joint_policies[0])):
                averaged_policy[i] = np.prod(
                    [joint_policy[i] for joint_policy in joint_policies], axis=0
                ) ** (1 / len(joint_policies))

            return averaged_policy

        elif self.population_averaging == "arithmetic":
            averaged_policy = [
                np.zeros_like(player_policy) for player_policy in joint_policies[0]
            ]

            for i in range(len(joint_policies[0])):
                averaged_policy[i] = np.mean(
                    [joint_policy[i] for joint_policy in joint_policies], axis=0
                )

            return averaged_policy

        else:
            raise ValueError(
                f"Unknown population_averaging method {self.population_averaging}"
            )
