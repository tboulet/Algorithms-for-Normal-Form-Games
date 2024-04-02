import random
import numpy as np
from typing import Any, Dict, List, Optional
from core.typing import JointPolicy, Policy
from algorithms.base_nfg_algorithm import BaseNFGAlgorithm
from core.metrics import get_distance_function


class PopulationBasedAlgorithm(BaseNFGAlgorithm):
    """A util class for all population based algorithms"""

    def __init__(self, sampler_population: dict, population_averaging: str) -> None:
        self.population_averaging = population_averaging
        self.sampler_population = sampler_population
        self.population = []
        self.kept_policies = []

    def sample_policies(self) -> List[JointPolicy]:
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

        sampling_pop_method = self.sampler_population["method"]
        if sampling_pop_method == "random":
            if self.sampler_population["distribution"] == "uniform":
                return random.sample(candidates_policies, size_population)
            elif self.sampler_population["distribution"] == "exponential":
                c = len(candidates_policies)
                weights = [2 ** (2 * x / c) for x in range(c)]
                return random.choices(
                    candidates_policies, weights=weights, k=size_population
                )

        elif sampling_pop_method == "periodic":
            return candidates_policies[:: n_last_policies_candidates // size_population]

        elif sampling_pop_method == "last":
            return [self.population[-1]]

        elif sampling_pop_method == "greedy":
            distance = self.sampler_population["distance"]
            distance_function = get_distance_function(distance)

            candidates_policies = self.population[-n_last_policies_candidates:]
            sampled_policies = [candidates_policies.pop()]

            greedy_mode = self.sampler_population["greedy_mode"]

            if greedy_mode == "to_average":
                for _ in range(size_population - 1):
                    average_policy = self.average_list_of_joint_policies(
                        sampled_policies
                    )
                    distance_list = [
                        distance_function(average_policy, policy)
                        for policy in candidates_policies
                    ]

                    max_index = distance_list.index(max(distance_list))
                    sampled_policies.append(candidates_policies.pop(max_index))

                return sampled_policies

            elif greedy_mode == "to_added_distance":
                distance_list = [0] * len(candidates_policies)
                for _ in range(size_population - 1):
                    for i, policy in enumerate(candidates_policies):
                        distance_list[i] += distance_function(
                            sampled_policies[-1],
                            policy,
                        )
                    max_index = distance_list.index(max(distance_list))

                    sampled_policies.append(candidates_policies.pop(max_index))
                    distance_list.pop(max_index)

                return sampled_policies

        else:
            raise ValueError(
                f"Unknown population_sampling method {sampling_pop_method}"
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
