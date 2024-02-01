from matplotlib import pyplot as plt
import numpy as np
from typing import Any, List, Callable, Tuple

from algorithms.base_nfg_algorithm import BaseNFGAlgorithm
from games.base_nfg_game import BaseNFGGame
from core.typing import JointPolicy, Policy
from core.utils import to_numeric



class IteratedForel(BaseNFGAlgorithm):
    def __init__(self,
        # FoReL specific parameters
        q_value_estimation_method : str,
        dynamics_method : str,
        learning_rate_rd : float,
        learning_rate_cum_values : float,
        n_monte_carlo_q_evaluation: int,
        regularizer: str,
        # Iterated FoReL specific parameters
        n_timesteps_per_iterations: int,
        eta: float,
    ) -> None:
        """Initializes the Iterated FoReL algorithm.

        Args:
            q_value_estimation_method (str): the method used to estimate the Q values (either "mc" or "model-based")
            dynamics_method (str): the method used to update the policy (either "softmax" or "rd" (Replicator Dynamics))
            learning_rate_rd (float): the learning rate used to update the policy (only used if dynamics_method == "rd")
            learning_rate_cum_values (float): the learning rate used to update the cumulative values (only used if dynamics_method == "softmax")
            n_monte_carlo_q_evaluation (int): the number of episodes used to estimate the Q values
            regularizer (str): the regularizer function tag (for now either "entropy" or "l2")
            n_timesteps_per_iterations (int): the number of timesteps per iteration
            eta (float): the eta parameter of the algorithm for modifying the rewards
        """
        self.q_value_estimation_method = q_value_estimation_method
        self.dynamics_method = dynamics_method
        self.learning_rate_rd = learning_rate_rd
        self.learning_rate_cum_values = learning_rate_cum_values
        self.n_timesteps_per_iterations = n_timesteps_per_iterations
        self.n_monte_carlo_q_evaluation = n_monte_carlo_q_evaluation
        self.eta = eta
        self.regularizer = regularizer
        

    # Interface methods
    
    def initialize_algorithm(self,
        game: BaseNFGGame,
        ) -> None:
        self.game = game
        self.n_actions = game.num_distinct_actions()
        self.n_players = game.num_players()
        
        self.joint_policy_mu = self.initialize_randomly_joint_policy(n_players=self.n_players, n_actions=self.n_actions)  # mu[i][a] = mu_i(a)
        self.joint_policy_pi = self.initialize_randomly_joint_policy(n_players=self.n_players, n_actions=self.n_actions)
        self.joint_cumulative_values = np.zeros((self.n_players, self.n_actions))
        self.joint_q_values = np.zeros((self.n_players, self.n_actions))
        
        self.iteration : int = 0
        self.timestep : int = 0
        self.monte_carlo_q_evaluation_episode_idx : int = 0
        

    def choose_joint_action(self,
        ) -> Tuple[List[int], List[float]]:
        return self.sample_joint_action_probs_from_policy(joint_policy=self.joint_policy_pi)
    
    
    def learn(self,
        joint_action: List[int],
        probs: List[float],
        rewards: List[float],
        ) -> None:
        
        # Those two booleans control which part of the algorithm is executed
        has_estimated_q_values = False
        has_done_forel_optimization = False
        
        
        # --- Modify the rewards ---
        returns_modified = self.modify_rewards(
            returns=rewards,
            chosen_actions=joint_action,
            pi=self.joint_policy_pi,
            mu=self.joint_policy_mu,
            eta=self.eta,
        )
        
        
        # --- Estimate Q values ---
        if self.q_value_estimation_method == "mc":
            # Method 1 : MC evaluation
            for i in range(self.n_players):
                self.joint_q_values[i][joint_action[i]] += returns_modified[i] / self.n_monte_carlo_q_evaluation  # Q^i_t(a) = Q^i_{t-1}(a) + r^i_t(a) / n_monte_carlo_q_evaluation
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
            
            # Increment timestep and reset the Q values
            self.timestep += 1
            self.joint_q_values = np.zeros((self.n_players, self.n_actions))
            if self.timestep == self.n_timesteps_per_iterations:
                self.timestep = 0
                has_done_forel_optimization = True
                
                
        # --- Update the mu policy ---
        if has_done_forel_optimization:
            # Set mu policy as the obtained pi policy, reset cumulative values
            self.iteration += 1
            self.joint_policy_mu = self.joint_policy_pi.copy()
            self.joint_cumulative_values = np.zeros((self.n_players, self.n_actions))
            self.joint_q_values = np.zeros((self.n_players, self.n_actions))

        
    
    def get_inference_policies(self,
        ) -> JointPolicy:
        return self.joint_policy_pi
    
    
    
    # Helper methods        
                    
    def modify_rewards(self, 
                        returns: List[float],
                        chosen_actions: List[int],
                        pi: JointPolicy,
                        mu: JointPolicy,
                        eta: float,
                    ) -> List[float]:
        """Implements the modification of rewards for the Forel algorithm.

        Args:
            returns (List[float]): the rewards obtained by the players
            chosen_actions (List[int]): the actions chosen by the players
            pi (JointPolicy): the joint policy used to choose the actions
            mu (JointPolicy): the regularization joint policy
            eta (float): a parameter of the algorithm

        Returns:
            List[float]: the modified rewards
        """
        if eta == 0:
            return returns
        
        n_players = len(pi)
        n_actions = len(pi[0])
        
        for i in range(n_players):
            pi_i_a = pi[i][chosen_actions[i]]
            pi_minus_i_a = np.prod([pi[j][chosen_actions[j]] for j in range(n_players) if j != i])
            mu_i_a = mu[i][chosen_actions[i]]
            mu_minus_i_a = np.prod([mu[j][chosen_actions[j]] for j in range(n_players) if j != i])
            returns[i] = returns[i] - eta * np.log(pi_i_a / mu_i_a) + eta * np.log(pi_minus_i_a / mu_minus_i_a)
        return returns
        
        
    def optimize_regularized_objective_function(self, 
            cum_values : List[List[float]],
            regularizer : str,
                ) -> Policy:
        """Apply the optimization step of the FoReL algorithm.

        Args:
            cum_values (List[List[float]]): the cumulative Q values of the players from the beginning of the episode
            regularizer (str): the regularizer function tag

        Returns:
            Policy: an agent-policy that is the result of the optimization step
        """
        if regularizer == "entropy":
            exp_cum_values = np.exp(cum_values)
            policy = exp_cum_values / np.sum(exp_cum_values)
            return policy
        elif regularizer == "l2":
            raise NotImplementedError
        else:
            raise NotImplementedError
            
