from matplotlib import pyplot as plt
import numpy as np
from typing import Any, List, Callable, Tuple, Dict, Optional

from algorithms.base_nfg_algorithm import BaseNFGAlgorithm
from games.base_nfg_game import BaseNFGGame
from core.typing import JointPolicy, Policy



class Forel(BaseNFGAlgorithm):
    def __init__(self,
        q_value_estimation_method : str,
        dynamics_method : str,
        learning_rate_rd : float,
        learning_rate_cum_values : float,
        n_monte_carlo_q_evaluation: int,
        regularizer: str,
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
        

    # Interface methods
    
    def initialize_algorithm(self,
        game: BaseNFGGame,
        ) -> None:
        self.game = game
        self.n_actions = game.num_distinct_actions()
        self.n_players = game.num_players()
        
        self.joint_policy_pi = self.initialize_randomly_joint_policy(n_players=self.n_players, n_actions=self.n_actions)
        self.joint_cumulative_values = np.zeros((self.n_players, self.n_actions))
        self.joint_q_values = np.zeros((self.n_players, self.n_actions))
        self.joint_count_seen_actions = np.zeros((self.n_players, self.n_actions))
        
        self.timestep : int = 0
        self.monte_carlo_q_evaluation_episode_idx : int = 0
        

    def choose_joint_action(self,
        ) -> Tuple[List[int], List[float]]:
        # Choose actions for both players
        return self.sample_joint_action_probs_from_policy(joint_policy=self.joint_policy_pi)
    
    
    def learn(self,
        joint_action: List[int],
        probs: List[float],
        rewards: List[float],
        ) -> Optional[Dict[str, float]]:
        
        # Those two booleans control which part of the algorithm is executed
        has_estimated_q_values = False
        
        # --- Estimate Q values ---
        if self.q_value_estimation_method == "mc":
            # Method 1 : MC evaluation
            # Incremental update of the Q values
            for i in range(self.n_players):
                if self.joint_count_seen_actions[i][joint_action[i]] == 0:
                    self.joint_q_values[i][joint_action[i]] = rewards[i]
                else:
                    self.joint_q_values[i][joint_action[i]] += (rewards[i] - self.joint_q_values[i][joint_action[i]]) / (self.joint_count_seen_actions[i][joint_action[i]] + 2)
                self.joint_count_seen_actions[i][joint_action[i]] += 1
            # Increment monte carlo q evaluation episode index
            self.monte_carlo_q_evaluation_episode_idx += 1
            if self.monte_carlo_q_evaluation_episode_idx == self.n_monte_carlo_q_evaluation:
                self.monte_carlo_q_evaluation_episode_idx = 0
                has_estimated_q_values = True
                
        elif self.q_value_estimation_method == "model-based":
            # Method 2 : get Q values from the game object (model-based)
            for i in range(self.n_players):
                for a in range(self.n_actions):
                    self.joint_q_values[i][a] = self.get_model_based_q_value(
                                                            game=self.game, 
                                                            joint_policy=self.joint_policy_pi, 
                                                            player=i, 
                                                            action=a,
                                                            )
            has_estimated_q_values = True
            
        
        # --- Update the policy ---
        if has_estimated_q_values:
            
            if self.dynamics_method == "softmax":
                # Method 1 : pi_i = softmax(cum_values_i)
                for i in range(self.n_players):
                    self.joint_cumulative_values[i] += self.learning_rate_cum_values * self.joint_q_values[i]
                for i in range(self.n_players):
                    self.joint_policy_pi[i] = self.optimize_regularized_objective_function(
                        cum_values=self.joint_cumulative_values[i],
                        regularizer=self.regularizer,
                    )
            
            elif self.dynamics_method == "rd":
                # Method 2 : Replicator Dynamics
                state_values = np.sum(self.joint_q_values * self.joint_policy_pi, axis=1)  # V_t = sum_a Q_t(a) * pi_t(a)
                advantage_values = self.joint_q_values - state_values[:, None]  # A_t(a) = Q_t(a) - V_t
                for i in range(self.n_players):
                    for a in range(self.n_actions):
                        self.joint_policy_pi[i][a] += self.learning_rate_rd * advantage_values[i][a] * self.joint_policy_pi[i][a]
                    # Normalize policy in case of numerical errors
                    self.joint_policy_pi[i] /= np.sum(self.joint_policy_pi[i])
            
            # Increment timestep and reset the Q values and count seen actions
            self.joint_q_values = np.zeros((self.n_players, self.n_actions))
            self.joint_count_seen_actions = np.zeros((self.n_players, self.n_actions))
            self.timestep += 1
        
        return {
            **{f"Q_0(a={a})" : self.joint_q_values[0][a] for a in range(self.n_actions)},
            **{f"y_0(a={a})" : self.joint_cumulative_values[0][a] for a in range(self.n_actions)},
            **{f"pi_0(a={a})" : self.joint_policy_pi[0][a] for a in range(self.n_actions)},
        }
    
    def get_inference_policies(self,
        ) -> JointPolicy:
        return self.joint_policy_pi
     
     
    # Helper methods
 
        
    def optimize_regularized_objective_function(self, 
            cum_values : List[List[float]],
            regularizer : str,
                ) -> Policy:
        """Apply dynamics

        Args:
            cum_values (List[List[float]]): the cumulative Q values of the players from the beginning of the episode
            regularizer (str): the regularizer function tag

        Returns:
            Policy: an agent-policy that is the result of the optimization step
        """
        if regularizer == "entropy":
            return self.get_softmax_policy_from_logits(logits=cum_values)
        elif regularizer == "l2":
            raise NotImplementedError
        else:
            raise NotImplementedError
    
    
