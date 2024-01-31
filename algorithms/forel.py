from matplotlib import pyplot as plt
import numpy as np
from typing import Any, List, Callable, Tuple

from algorithms.base_nfg_algorithm import BaseNFGAlgorithm
from games.base_nfg_game import BaseNFGGame
from core.typing import JointPolicy, Policy



class Forel(BaseNFGAlgorithm):
    def __init__(self,
        q_value_estimation_method : str,
        learning_rate : float,
        n_monte_carlo_q_evaluation: int,
        regularizer: str,
    ) -> None:
        """Initializes the Follow the Regularized Leader (FoReL) algorithm.
        This algorithm try to maximize an exploitation term that consist of a (potentially weighted) sum of the Q values and
        exploration terms that are the regularizers (e.g. the entropy)

        Args:
            q_value_estimation_method (str): the method used to estimate the Q values (either "mc" or "model-based")
            learning_rate (float): the learning rate used to update the policy
            n_monte_carlo_q_evaluation (int): the number of episodes used to estimate the Q values
            regularizer (str): the regularizer function tag (for now either "entropy" or "l2")
        """
        self.q_value_estimation_method = q_value_estimation_method
        self.learning_rate = learning_rate
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
        ) -> None:
        
        
        # Update cumulative Q values
        has_estimated_q_values = False
        
        
        if self.q_value_estimation_method == "mc":
            # #Method 1 : MC evaluation
            for i in range(self.n_players):
                self.joint_q_values[i][joint_action[i]] += rewards[i] / self.n_monte_carlo_q_evaluation  # Q^i_t(a) = Q^i_{t-1}(a) + r^i_t(a) / n_monte_carlo_q_evaluation
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
            
        
        
        if has_estimated_q_values:
            # Update cumulative values and reset Q values
            lr = 1  # TODO: check if we should use a learning rate
            for i in range(self.n_players):
                self.joint_cumulative_values[i] += lr * self.joint_q_values[i]
            
            # Update pi by optimizing the regularized objective function
            # for i in range(self.n_players):
            #     self.joint_policy_pi[i] = self.optimize_regularized_objective_function(
            #         cum_values=self.joint_cumulative_values[i],
            #         regularizer=self.regularizer,
            #     )
            
            # Replicator Dynamics
            state_values = np.sum(self.joint_q_values * self.joint_policy_pi, axis=1)  # V_t = sum_a Q_t(a) * pi_t(a)
            advantage_values = self.joint_q_values - state_values[:, None]  # A_t(a) = Q_t(a) - V_t
            for i in range(self.n_players):
                for a in range(self.n_actions):
                    self.joint_policy_pi[i][a] += self.learning_rate * advantage_values[i][a] * self.joint_policy_pi[i][a]
                # Normalize policy in case of numerical errors
                self.joint_policy_pi[i] /= np.sum(self.joint_policy_pi[i])
            
            # Increment timestep
            self.joint_q_values = np.zeros((self.n_players, self.n_actions))
            self.timestep += 1
    
    
    def do_stop_learning(self,
        ) -> bool:
        return False
    
    
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
    
    
