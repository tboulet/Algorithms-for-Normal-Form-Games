import random
import sys
import numpy as np
from typing import Any, Dict, List, Optional

from algorithms.forel import Forel
from core.online_plotter import DataPolicyToPlot
from core.scheduler import Scheduler
from core.utils import instantiate_class, to_numeric, try_get
from games.base_nfg_game import BaseNFGGame
from core.typing import JointPolicy, Policy
from copy import deepcopy


class PopulationAlternatingLyapunovForel(Forel):
    def __init__(
        self,
        # FoReL specific parameters
        forel_config: Dict[str, Any],
        # Population Iterated Lyapunov FoReL specific parameters
        n_timesteps_pc_phase: int,
        n_timesteps_lyapunov_phase: int,
        population_averaging: str,
        sampler_population: Dict[str, Any],
        eta_scheduler_config: dict,
    ) -> None:
        """Initialize the Population Alternating Lyapunov FoReL algorithm.
        This algorithm alternates between a Pointcarré (PC) phase, where it cycles around the Nash Equilibrium,
        and a Lyapunov phase, where the reward are regularized by a regularization term that depends on the regularized policy mu.

        In particular, mu is selected as the population's average policy of the last PC phase.

        """
        super().__init__(**forel_config)
        self.n_timesteps_pc_phase = n_timesteps_pc_phase
        self.n_timesteps_lyapunov_phase = n_timesteps_lyapunov_phase
        self.population_averaging = population_averaging
        self.sampler_population = sampler_population
        self.eta_scheduler : Scheduler = instantiate_class(config=eta_scheduler_config)
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
            assert len(self.population) == self.n_timesteps_pc_phase, f"len(self.population) = {len(self.population)} != self.n_timesteps_pc_phase = {self.n_timesteps_pc_phase}"
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
            self.step
            % (self.n_timesteps_pc_phase + self.n_timesteps_lyapunov_phase) < self.n_timesteps_pc_phase
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
        self.metrics["eta"] = self.get_eta()

        for i in range(self.n_players):
            self.metrics[f"reward_modif/reward_modif_{i}"] = rewards[i]

        self.step += 1
        return self.metrics

    # Helper methods
    def get_eta(self) -> float:
        return self.eta_scheduler.get_value(self.step)

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

    def sample_policies(self) -> List[JointPolicy]:
        sampling_pop_method = self.sampler_population["method"]
        if sampling_pop_method == "random":
            n_last_policies_candidates = self.sampler_population[
                "n_last_policies_to_sample"
            ]
            n_last_policies_candidates = (
                n_last_policies_candidates
                if n_last_policies_candidates is not None
                else len(self.population)
            )
            candidates_policies = self.population[-n_last_policies_candidates:]
            size_population = self.sampler_population["size_population"]
            if self.sampler_population["distribution"] == "uniform":
                return random.sample(candidates_policies, size_population)
            elif self.sampler_population["distribution"] == "exponential":
                c = len(candidates_policies)
                weights = [2 ** (2 * x / c) for x in range(c)]
                return random.choices(
                    candidates_policies, weights=weights, k=size_population
                )

        elif sampling_pop_method == "periodic":
            n_last_policies_candidates = self.sampler_population[
                "n_last_policies_to_sample"
            ]
            n_last_policies_candidates = (
                n_last_policies_candidates
                if n_last_policies_candidates is not None
                else len(self.population)
            )
            candidates_policies = self.population[-n_last_policies_candidates:]
            size_population = self.sampler_population["size_population"]
            return candidates_policies[:: n_last_policies_candidates // size_population]

        elif sampling_pop_method == "last":
            return [self.population[-1]]

        elif sampling_pop_method == "greedy":
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
