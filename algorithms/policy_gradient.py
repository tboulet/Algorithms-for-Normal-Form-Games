from matplotlib import pyplot as plt
import numpy as np
from typing import Any, List, Callable, Tuple

from algorithms.base_nfg_algorithm import BaseNFGAlgorithm
from games.base_nfg_game import BaseNFGGame
from core.typing import JointPolicy, Policy
from core.utils import to_numeric



class PolicyGradient(BaseNFGAlgorithm):
    def __init__(self,
        learning_rate : float,
        q_value_estimation_method : str,
        n_monte_carlo_q_evaluation : int,
    ) -> None:
        """Initializes the (Softmax) Policy Gradient algorithm.
        This algorithm try to maximize the expected reward by using a softmax tabular policy gradient approach.

        Args:
            learning_rate (float): the learning rate
            q_value_estimation_method (str): the method used to estimate the Q values
            n_monte_carlo_q_evaluation (int): the number of episodes used to estimate the Q values
        """
        self.learning_rate = learning_rate
        self.q_value_estimation_method = q_value_estimation_method
        self.n_monte_carlo_q_evaluation = n_monte_carlo_q_evaluation
        

    # Interface methods
    
    def initialize_algorithm(self,
        game: BaseNFGGame,
        ) -> None:
        self.game = game
        self.n_actions = game.num_distinct_actions()
        self.n_players = game.num_players()
        
        self.joint_logits = np.exp(self.initialize_randomly_joint_policy(n_players=self.n_players, n_actions=self.n_actions))
        self.joint_q_values = np.zeros((self.n_players, self.n_actions))
        
        self.timestep : int = 0
        self.monte_carlo_q_evaluation_episode_idx : int = 0
        

    def choose_joint_action(self,
        ) -> Tuple[List[int], List[float]]:
        joint_policy_pi = self.get_softmax_joint_policy_from_logits(joint_logits=self.joint_logits)
        return self.sample_joint_action_probs_from_policy(joint_policy=joint_policy_pi)
    
    
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
                                                            player=i, 
                                                            action=a, 
                                                            joint_policy=self.get_softmax_joint_policy_from_logits(joint_logits=self.joint_logits),
                                                            )
            has_estimated_q_values = True
            
        
        
        if has_estimated_q_values:
            
            # Update logits and reset Q values
            
            
            joint_policy = self.get_softmax_joint_policy_from_logits(joint_logits=self.joint_logits)
            state_values = np.sum(joint_policy * self.joint_q_values, axis=1)
            advantage_values = self.joint_q_values - state_values  # A_t(a) = Q_t(a) - V_t
            self.joint_logits = self.joint_logits + self.learning_rate * (1 - self.get_softmax_joint_policy_from_logits(joint_logits=self.joint_logits)) * advantage_values
            self.joint_q_values = np.zeros((self.n_players, self.n_actions))
            
            # Increment timestep
            self.timestep += 1
    
    
    def do_stop_learning(self,
        ) -> bool:
        return False
    
    
    def get_inference_policies(self,
        ) -> JointPolicy:
        return self.get_softmax_joint_policy_from_logits(joint_logits=self.joint_logits)
     
     
     
    
            

    
    
    
    
