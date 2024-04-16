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
        forel_config: Dict[str, any],
    ) -> None:
        """Initializes the Follow the Regularized Leader (FoReL) algorithm.
        This algorithm try to maximize an exploitation term that consist of a (potentially weighted) sum of the Q values and
        exploration terms that are the regularizers (e.g. the entropy)

        Args:
            forel_config (Dict[str, any]): the configuration of the FoReL algorithm. It should contain the following keys:
                dynamics_method (Dict[str, any]): the configuration of the dynamics method.
                q_value_estimation_method (Dict[str, any]): the configuration of the Q value estimation method.
        """
        self.forel_config = forel_config
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
        # Batch update : Q_t(a) = sum_{t=1}^{T} r_t
        for i in range(self.n_players):
            if self.joint_count_seen_actions[i][joint_action[i]] == 0:
                self.joint_q_values[i][joint_action[i]] = rewards[i]
            else:
                self.joint_q_values[i][joint_action[i]] += rewards[i]
            self.joint_count_seen_actions[i][joint_action[i]] += 1
        # Increment monte carlo q evaluation episode index
        self.monte_carlo_q_evaluation_episode_idx += 1
        n_monte_carlo_q_evaluation = self.forel_config["q_value_estimation_method"][
            "n_monte_carlo_q_evaluation"
        ]
        if self.monte_carlo_q_evaluation_episode_idx == n_monte_carlo_q_evaluation:
            self.monte_carlo_q_evaluation_episode_idx = 0
            for i in range(self.n_players):
                for a in range(self.n_actions[i]):
                    if self.joint_count_seen_actions[i][a] > 0:
                        self.joint_q_values[i][a] /= self.joint_count_seen_actions[i][a]
            return True

        return False

    def transform_q_value(self) -> None:
        raise ValueError("This function only exists for forel lyapunov")

    def compute_model_based_q_values(self) -> None:
        for i in range(self.n_players):
            self.joint_q_values[i] = self.game.get_model_based_q_value(
                player=i,
                joint_policy=self.joint_policy_pi,
            )
        if self.lyapunov:
            self.transform_q_value()

    def update_policy(self):
        dynamics_method = self.forel_config["dynamics_method"]["method"]
        if dynamics_method == DynamicMethod.SOFTMAX:
            # Method 1 : pi_i = softmax(cum_values_i)
            learning_rate_cum_values = self.forel_config["dynamics_method"][
                "learning_rate_cum_values"
            ]
            regularizer = self.forel_config["dynamics_method"]["regularizer"]
            for i in range(self.n_players):
                self.joint_cumulative_values[i] += (
                    learning_rate_cum_values * self.joint_q_values[i]
                )
                self.joint_policy_pi[i] = self.optimize_regularized_objective_function(
                    cum_values=self.joint_cumulative_values[i],
                    regularizer=regularizer,
                )

        elif dynamics_method == DynamicMethod.RD:
            # Method 2 : Replicator Dynamics
            learning_rate_rd = self.forel_config["dynamics_method"]["learning_rate_rd"]
            for i in range(self.n_players):

                state_value = np.sum(
                    self.joint_q_values[i] * self.joint_policy_pi[i]
                )  # V_t = sum_a Q_t(a) * pi_t(a)
                advantage_values = (
                    self.joint_q_values[i] - state_value
                )  # A_t(a) = Q_t(a) - V_t
                self.joint_policy_pi[i] += (
                    learning_rate_rd * advantage_values * self.joint_policy_pi[i]
                )
                # Normalize policy in case of numerical errors
                self.joint_policy_pi[i] /= np.sum(self.joint_policy_pi[i])

        else:
            raise ValueError(f"Unknown dynamics_method : {dynamics_method}")

    def learn(
        self,
        joint_action: List[int],
        probs: List[float],
        rewards: List[float],
    ) -> Optional[Dict[str, float]]:

        # Those two booleans control which part of the algorithm is executed
        has_estimated_q_values = False

        # --- Estimate Q values ---
        q_value_estimation_method = self.forel_config["q_value_estimation_method"][
            "method"
        ]
        if q_value_estimation_method == QValueEstimationMethod.MONTE_CARLO_BATCHED:
            # Batch MC evaluation : Q_t(a) = sum_{t=1}^{T} r_t
            has_estimated_q_values = self.monte_carlo_step(
                joint_action=joint_action, rewards=rewards
            )

        elif q_value_estimation_method == QValueEstimationMethod.MONTE_CARLO_INCREMENTAL:
            # Incremental MC evaluation : Q_t(a) += alpha * (r_t - Q_t(a))
            learning_rate_q_values = self.forel_config["q_value_estimation_method"][
                "learning_rate_q_values"
            ]
            for i in range(self.n_players):
                if self.joint_count_seen_actions[i][joint_action[i]] == 0:
                    self.joint_q_values[i][joint_action[i]] = rewards[i]
                else:
                    self.joint_q_values[i][
                        joint_action[i]
                    ] += learning_rate_q_values * (
                        rewards[i] - self.joint_q_values[i][joint_action[i]]
                    )
                self.joint_count_seen_actions[i][joint_action[i]] += 1
            has_estimated_q_values = True

        elif q_value_estimation_method == QValueEstimationMethod.MODEL_BASED:
            # Model-based extraction of Q values
            self.compute_model_based_q_values()
            has_estimated_q_values = True

        else:
            raise ValueError(
                f"Unknown q_value_estimation_method : {q_value_estimation_method}"
            )

        # --- Update the policy ---
        if has_estimated_q_values:
            self.update_policy()

            # Increment timestep and in MC batch, reset the Q values and count seen actions
            if q_value_estimation_method == QValueEstimationMethod.MONTE_CARLO_BATCHED:
                self.joint_q_values = [
                    np.zeros(n_action) for n_action in self.n_actions
                ]
                self.joint_count_seen_actions = [
                    np.zeros(n_action) for n_action in self.n_actions
                ]
            self.timestep += 1

        objects_to_log = {}
        for i in range(self.n_players):
            for a in range(self.n_actions[i]):
                if not (i == 0 and a in [0, 1]):
                    continue  # only log X_0(a=0, 1) to limit the number of logs
                objects_to_log[f"Q_{i}/Q_{i}(a={a})"] = self.joint_q_values[i][a]
                objects_to_log[f"pi_{i}/pi_{i}(a={a})"] = self.joint_policy_pi[i][a]
                objects_to_log[f"cum_values_{i}/cum_values_{i}(a={a})"] = (
                    self.joint_cumulative_values[i][a]
                )

        return objects_to_log

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
