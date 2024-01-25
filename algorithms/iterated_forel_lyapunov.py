from matplotlib import pyplot as plt
import numpy as np
from typing import Any, List, Callable, Tuple

from algorithms.base_nfg_algorithm import BaseNFGAlgorithm
from games.base_nfg_game import BaseNFGGame
from core.typing import JointPolicy, Policy
from core.utils import to_numeric



class IteratedForel(BaseNFGAlgorithm):
    def __init__(self,
        n_iterations_max : int,
        n_timesteps_per_iterations: int,
        n_monte_carlo_q_evaluation: int,
        eta: float,
        regularizer: str,
    ) -> None:
        """Initializes the Iterated FoReL algorithm.

        Args:
            n_iterations_max (int): the maximum number of iterations
            n_timesteps_per_iterations (int): the number of timesteps per iteration
            n_monte_carlo_q_evaluation (int): the number of episodes used to estimate the Q values
            eta (float): the eta parameter of the algorithm
            regularizer (str): the regularizer function tag (for now either "entropy" or "l2")
        """
        self.n_iterations_max = to_numeric(n_iterations_max)
        self.n_timesteps_per_iterations = n_timesteps_per_iterations
        self.n_monte_carlo_q_evaluation = n_monte_carlo_q_evaluation
        self.eta = eta
        self.regularizer = regularizer
        

    # Interface methods
    
    def initialize_algorithm(self,
        game: BaseNFGGame,
        ) -> None:
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
        
        
        # Modify the rewards
        returns_modified = self.modify_rewards(
            returns=rewards,
            chosen_actions=joint_action,
            pi=self.joint_policy_pi,
            mu=self.joint_policy_mu,
            eta=self.eta,
        )
        
        # Update cumulative Q values
        for i in range(self.n_players):
            self.joint_q_values[i][joint_action[i]] += returns_modified[i] / self.n_monte_carlo_q_evaluation  # Q^i_t(a) = Q^i_{t-1}(a) + r^i_t(a) / n_monte_carlo_q_evaluation

        # Increment monte carlo q evaluation episode index
        self.monte_carlo_q_evaluation_episode_idx += 1
        if self.monte_carlo_q_evaluation_episode_idx == self.n_monte_carlo_q_evaluation:
            self.monte_carlo_q_evaluation_episode_idx = 0
            
            # Update cumulative values and reset Q values
            lr = 1  # TODO: check if we should use a learning rate
            for i in range(self.n_players):
                self.joint_cumulative_values[i] += lr * self.joint_q_values[i]
            self.joint_q_values = np.zeros((self.n_players, self.n_actions))
            
            # Update pi by optimizing the regularized objective function
            for i in range(self.n_players):
                self.joint_policy_pi[i] = self.optimize_regularized_objective_function(
                    cum_values=self.joint_cumulative_values[i],
                    regularizer=self.regularizer,
                )
                
            # Increment iteration index
            self.timestep += 1
            if self.timestep == self.n_timesteps_per_iterations:
                self.timestep = 0
                # Set mu policy as the obtained pi policy, reset cumulative values
                self.iteration += 1
                self.joint_policy_mu = self.joint_policy_pi.copy()
                self.joint_cumulative_values = np.zeros((self.n_players, self.n_actions))
                self.joint_q_values = np.zeros((self.n_players, self.n_actions))

            if self.iteration == self.n_iterations_max:
                exit()
    
    
    def do_stop_learning(self,
        ) -> bool:
        return self.iteration >= self.n_iterations_max
    
    
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
        """Apply dynamics

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
            
