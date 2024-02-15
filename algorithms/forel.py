from matplotlib import pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Optional

from algorithms.base_nfg_algorithm import BaseNFGAlgorithm
from games.base_nfg_game import BaseNFGGame
from core.typing import (
    JointPolicy,
    Policy,
    DynamicMethod,
    QValueEstimationMethod,
    Regularizer,
)


class Forel(BaseNFGAlgorithm):
    def __init__(
        self,
        q_value_estimation_method: QValueEstimationMethod,
        dynamics_method: DynamicMethod,
        learning_rate_rd: float,
        learning_rate_cum_values: float,
        n_monte_carlo_q_evaluation: int,
        regularizer: Regularizer,
    ) -> None:
        """Initializes the Follow the Regularized Leader (FoReL) algorithm.
        This algorithm try to maximize an exploitation term that consist of a (potentially weighted) sum of the Q values and
        exploration terms that are the regularizers (e.g. the entropy)

        Args:
            q_value_estimation_method (str): the method used to estimate the Q values (either "mc" or "model-based")
            dynamics_method (str): the method used to update the policy (either "softmax" or "rd" (Replicator Dynamics))
            learning_rate_rd (float): the learning rate used to update the policy (only used if dynamics_method == "rd")
            learning_rate_cum_values (float): the learning rate used to update the cumulative values (only used if dynamics_method == "softmax")
            n_monte_carlo_q_evaluation (int): the number of episodes used to estimate the Q values
            regularizer (str): the regularizer function tag (for now either "entropy" or "l2")
        """
        self.q_value_estimation_method = q_value_estimation_method
        self.dynamics_method = dynamics_method
        self.learning_rate_rd = learning_rate_rd
        self.learning_rate_cum_values = learning_rate_cum_values
        self.n_monte_carlo_q_evaluation = n_monte_carlo_q_evaluation
        self.regularizer = regularizer
        self.lyapunov = False

    # Interface methods

    def initialize_algorithm(
        self,
        game: BaseNFGGame,
        joint_policy_pi: Optional[JointPolicy] = None,
    ) -> None:
        self.game = game
        self.n_actions = game.num_distinct_actions()
        self.n_players = game.num_players()

        self.joint_policy_pi = (
            self.initialize_randomly_joint_policy(n_actions=self.n_actions)
            if joint_policy_pi is None
            else [np.array(policy) for policy in joint_policy_pi]
        )
        self.joint_cumulative_values = [
            np.zeros(n_action) for n_action in self.n_actions
        ]
        self.joint_q_values = [np.zeros(n_action) for n_action in self.n_actions]
        self.joint_count_seen_actions = [
            np.zeros(n_action) for n_action in self.n_actions
        ]

        self.timestep: int = 0
        self.monte_carlo_q_evaluation_episode_idx: int = 0

    def choose_joint_action(
        self,
    ) -> Tuple[List[int], List[float]]:
        # Choose actions for both players
        return self.sample_joint_action_probs_from_policy(
            joint_policy=self.joint_policy_pi
        )

    def monte_carlo_step(self, joint_action: List[int], rewards: List[float]) -> bool:
        """Update the Q values using Monte Carlo evaluation

        Args:
            joint_action (List[int]): the joint actions played
            rewards (List[float]): the rewards obtained by the players

        Returns:
            bool: True if the monte carlo evaluation is finished, False otherwise

        """
        for i in range(self.n_players):
            if self.joint_count_seen_actions[i][joint_action[i]] == 0:
                self.joint_q_values[i][joint_action[i]] = rewards[i]
            else:
                self.joint_q_values[i][joint_action[i]] += (
                    rewards[i] - self.joint_q_values[i][joint_action[i]]
                ) / (self.joint_count_seen_actions[i][joint_action[i]] + 1)
            self.joint_count_seen_actions[i][joint_action[i]] += 1

        # Increment monte carlo q evaluation episode index
        self.monte_carlo_q_evaluation_episode_idx += 1
        if self.monte_carlo_q_evaluation_episode_idx == self.n_monte_carlo_q_evaluation:
            self.monte_carlo_q_evaluation_episode_idx = 0
            return True

        return False

    def transform_q_value(self):
        raise ValueError("This function only exists for forel lyapunov")

    def compute_model_based_q_values(self):
        for i in range(self.n_players):
            self.joint_q_values[i] = self.game.get_model_based_q_value(
                player=i,
                joint_policy=self.joint_policy_pi,
            )
        if self.lyapunov:
            self.transform_q_value()

    def update_policy(self):
        if self.dynamics_method == DynamicMethod.SOFTMAX:
            # Method 1 : pi_i = softmax(cum_values_i)
            for i in range(self.n_players):
                self.joint_cumulative_values[i] += (
                    self.learning_rate_cum_values * self.joint_q_values[i]
                )
                self.joint_policy_pi[i] = self.optimize_regularized_objective_function(
                    cum_values=self.joint_cumulative_values[i],
                    regularizer=self.regularizer,
                )

        elif self.dynamics_method == DynamicMethod.RD:
            # Method 2 : Replicator Dynamics
            for i in range(self.n_players):
                state_value = np.sum(
                    self.joint_q_values[i] * self.joint_policy_pi[i]
                )  # V_t = sum_a Q_t(a) * pi_t(a)
                advantage_values = (
                    self.joint_q_values[i] - state_value
                )  # A_t(a) = Q_t(a) - V_t
                self.joint_policy_pi[i] += (
                    self.learning_rate_rd
                    * advantage_values[i]
                    * self.joint_policy_pi[i]
                )
                # Normalize policy in case of numerical errors
                self.joint_policy_pi[i] /= np.sum(self.joint_policy_pi[i])

    def learn(
        self,
        joint_action: List[int],
        probs: List[float],
        rewards: List[float],
    ) -> Optional[Dict[str, float]]:

        # Those two booleans control which part of the algorithm is executed
        has_estimated_q_values = False

        # --- Estimate Q values ---
        if self.q_value_estimation_method == QValueEstimationMethod.MC:
            # Method 1 : MC evaluation
            has_estimated_q_values = self.monte_carlo_step(
                joint_action=joint_action, rewards=rewards
            )

        elif self.q_value_estimation_method == QValueEstimationMethod.MODEL_BASED:
            # Method 2 : Model-based exact evaluation
            self.compute_model_based_q_values()
            has_estimated_q_values = True

        else:
            raise ValueError(
                f"Unknown q_value_estimation_method : {self.q_value_estimation_method}"
            )

        # --- Update the policy ---
        if has_estimated_q_values:
            self.update_policy()

            # Increment timestep and reset the Q values and count seen actions
            self.joint_q_values = [np.zeros(n_action) for n_action in self.n_actions]
            self.joint_count_seen_actions = [
                np.zeros(n_action) for n_action in self.n_actions
            ]
            self.timestep += 1

        return {
            **{f"Q_{i}(a=0)": self.joint_q_values[i][0] for i in range(self.n_players)},
            **{
                f"y_0(a={a})": self.joint_cumulative_values[0][a]
                for a in range(self.n_actions)
            },
            **{
                f"pi_0(a={a})": self.joint_policy_pi[0][a]
                for a in range(self.n_actions)
            },
        }

    def get_inference_policies(
        self,
    ) -> JointPolicy:
        return self.joint_policy_pi

    # Helper methods

    def optimize_regularized_objective_function(
        self,
        cum_values: List[List[float]],
        regularizer: Regularizer,
    ) -> Policy:
        """Apply dynamics

        Args:
            cum_values (List[List[float]]): the cumulative Q values of the players from the beginning of the episode
            regularizer (str): the regularizer function tag

        Returns:
            Policy: an agent-policy that is the result of the optimization step
        """
        if regularizer == Regularizer.ENTROPY:
            return self.get_softmax_policy_from_logits(logits=cum_values)
        elif regularizer == Regularizer.L2:
            raise NotImplementedError("L2 regularizer not implemented yet")
        else:
            raise NotImplementedError(f"Unknown regularizer function{regularizer}")
