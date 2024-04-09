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


class PopulationAlternatingLyapunovForel(
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
        # PAL-FoReL specific parameters
        n_timesteps_pc_phase: int,
        n_timesteps_lyapunov_phase: int,
        eta_scheduler_config: dict,
    ) -> None:
        """Initialize the Population Alternating Lyapunov FoReL algorithm.
        This algorithm alternates between a Pointcarré (PC) phase, where it cycles around the Nash Equilibrium,
        and a Lyapunov phase, where the reward are regularized by a regularization term that depends on the regularized policy mu.

        In particular, mu is selected as the population's average policy of the last PC phase.

        """
        Forel.__init__(self, **forel_config)
        PopulationBasedAlgorithm.__init__(
            self,
            sampler_population=sampler_population,
            population_averaging=population_averaging,
            n_last_policies_to_sample=n_last_policies_to_sample,
        )
        LyapunovBasedAlgorithm.__init__(self)

        self.n_timesteps_pc_phase = n_timesteps_pc_phase
        self.n_timesteps_lyapunov_phase = n_timesteps_lyapunov_phase
        self.eta_scheduler: Scheduler = instantiate_class(config=eta_scheduler_config)
        self.lyapunov = False

    def initialize_algorithm(
        self,
        game: BaseNFGGame,
        joint_policy_pi: Optional[JointPolicy] = None,
    ) -> None:
        self.metrics = {}
        super().initialize_algorithm(game=game, joint_policy_pi=joint_policy_pi)
        self.iteration: int = 0
        self.step: int = 0

    def learn(
        self,
        joint_action: List[int],
        probs: List[float],
        rewards: List[float],
    ) -> None:

        # --- When we have reached the end of a Lyapunov phase, we go back to non-lyapunov mode ---
        if (
            self.step % (self.n_timesteps_pc_phase + self.n_timesteps_lyapunov_phase)
            == 0
        ):
            print("End of Lyapunov phase. PC phase starting.")
            self.population = []
            self.kept_policies = []
            self.lyapunov = False

        # --- When we reach the end of a PC phase, we sample the policies among the population of the last PC phase ---
        elif (
            self.step % (self.n_timesteps_pc_phase + self.n_timesteps_lyapunov_phase)
            == self.n_timesteps_pc_phase
        ):
            print("End of PC phase. Lyapunov phase starting.")
            self.kept_policies = self.sample_policies()
            self.joint_policy_mu = self.average_list_of_joint_policies(
                self.kept_policies
            )
            super().initialize_algorithm(
                game=self.game,
                joint_policy_pi=self.joint_policy_pi,
            )
            self.lyapunov = True

            # Add dataPolicies to plot
            self.metrics["pi_sampled"] = DataPolicyToPlot(
                name="π_sampled",
                joint_policy=self.kept_policies,
                color="black",
                marker="o",
                is_unique=True,
            )
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

        # Check in which phase we are
        is_pc_phase = (
            self.step % (self.n_timesteps_pc_phase + self.n_timesteps_lyapunov_phase)
            < self.n_timesteps_pc_phase
        )

        # PC phase : we don't modify the rewards and we keep the population in memory
        if is_pc_phase:
            metrics_forel = super().learn(
                joint_action=joint_action, probs=probs, rewards=rewards
            )
            self.metrics.update(metrics_forel)
            self.population.append(deepcopy(self.joint_policy_pi))

        # Lyapunov phase : we modify the rewards and we don't keep the population in memory
        else:
            rewards += self.lyapunov_reward(
                chosen_actions=joint_action,
                pi=self.joint_policy_pi,
                mu=self.joint_policy_mu,
            )
            metrics_forel = super().learn(
                joint_action=joint_action, probs=probs, rewards=rewards
            )
            self.metrics.update(metrics_forel)

        # Add the metrics and dataPolicies to plot
        self.metrics["timestep"] = self.timestep
        self.metrics["iteration"] = self.step // (
            self.n_timesteps_pc_phase + self.n_timesteps_lyapunov_phase
        )
        self.metrics["is_pc_phase"] = int(is_pc_phase)
        self.metrics["population_size"] = len(self.population)
        self.metrics["eta"] = self.get_eta()

        for i in range(self.n_players):
            self.metrics[f"reward_modif/reward_modif_{i}"] = rewards[i]

        self.step += 1
        return self.metrics

    # Helper methods
    def get_eta(self) -> float:
        return self.eta_scheduler.get_value(self.step)
