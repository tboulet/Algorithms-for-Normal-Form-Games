import sys
import numpy as np
from typing import Any, Dict, List, Optional
import random


from algorithms.forel import Forel
from core.online_plotter import DataPolicyToPlot
from core.scheduler import Scheduler
from core.utils import to_numeric
from games.base_nfg_game import BaseNFGGame
from core.typing import JointPolicy, Policy
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
        if self.timestep == self.population_timesteps_per_iterations:
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

    def sample_policies(self) -> List[JointPolicy]:
        if self.population_sampling == "random":
            return random.sample(self.population, self.population_k)

        elif self.population_sampling == "greedy":
            raise NotImplementedError("Greedy sampling not implemented yet")

        else:
            raise ValueError(
                f"Unknown population_sampling method {self.population_sampling}"
            )

    def average_list_of_joint_policies(
        self, list_joint_policies: List[JointPolicy]
    ) -> JointPolicy:
        """Agglomerates the joint policies using the population_averaging method.

        Args:
            list_joint_policies (List[JointPolicy]): a list of joint policies to agglomerate

        Returns:
            JointPolicy: the agglomerated joint policy
        """
        assert (
            len(list_joint_policies) > 0
        ), "The list of joint policies should not be empty"
        n_players = len(list_joint_policies[0])
        return [
            self.average_list_of_policies(
                [joint_policy[i] for joint_policy in list_joint_policies]
            )
            for i in range(n_players)
        ]

    def average_list_of_policies(self, list_policies: List[Policy]) -> Policy:
        """Agglomerates the policies using the population_averaging method.

        Args:
            list_policies (List[Policy]): a list of policies to agglomerate

        Returns:
            Policy: the agglomerated policy
        """
        if self.population_averaging == "geometric":
            n_policies = len(list_policies)
            averaged_policy = np.prod(list_policies, axis=0) ** (1 / n_policies)
            averaged_policy /= averaged_policy.sum()
            return averaged_policy

        elif self.population_averaging == "arithmetic":
            averaged_policy = np.mean(list_policies, axis=0)
            averaged_policy /= averaged_policy.sum()
            return averaged_policy

        else:
            raise ValueError(
                f"Unknown population_averaging method {self.population_averaging}"
            )
