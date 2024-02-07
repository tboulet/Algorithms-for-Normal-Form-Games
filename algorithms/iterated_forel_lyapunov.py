from time import sleep
from matplotlib import pyplot as plt
import numpy as np
from typing import Any, Dict, List, Callable, Tuple

from algorithms.base_nfg_algorithm import BaseNFGAlgorithm
from algorithms.forel import Forel
from games.base_nfg_game import BaseNFGGame
from core.typing import JointPolicy, Policy
from core.utils import to_numeric



class IteratedForel(Forel):
    def __init__(self,
        # FoReL specific parameters
        forel_config : Dict[str, Any],
        # Iterated FoReL specific parameters
        n_timesteps_per_iterations: int,
        eta: float,
    ) -> None:
        """Initializes the Iterated FoReL algorithm.

        Args:
            forel_config (Dict[str, Any]): the configuration of the FoReL algorithm. It should contain the following keys:
                - q_value_estimation_method (str): the method used to estimate the Q values (either "mc" or "model-based")
                - dynamics_method (str): the method used to update the policy (either "softmax" or "rd" (Replicator Dynamics))
                - learning_rate_rd (float): the learning rate used to update the policy (only used if dynamics_method == "rd")
                - learning_rate_cum_values (float): the learning rate used to update the cumulative values (only used if dynamics_method == "softmax")
                - n_monte_carlo_q_evaluation (int): the number of episodes used to estimate the Q values
                - regularizer (str): the regularizer function tag (for now either "entropy" or "l2")
            n_timesteps_per_iterations (int): the number of timesteps per iteration
            eta (float): the eta parameter of the algorithm for modifying the rewards
        """
        super().__init__(**forel_config)
        self.n_timesteps_per_iterations = n_timesteps_per_iterations
        self.eta = eta
        

    # Interface methods
    
    def initialize_algorithm(self,
        game: BaseNFGGame,
        ) -> None:
        super().initialize_algorithm(game=game)
        self.iteration : int = 0
        self.joint_policy_mu = self.initialize_randomly_joint_policy(n_players=self.n_players, n_actions=self.n_actions)
    
    
    def learn(self,
        joint_action: List[int],
        probs: List[float],
        rewards: List[float],
        ) -> None:
        
        
        # --- Modify the rewards ---
        returns_modified = self.modify_rewards(
            returns=rewards,
            chosen_actions=joint_action,
            pi=self.joint_policy_pi,
            mu=self.joint_policy_mu,
            eta=self.eta,
        )
        
        # --- Do one learning step of FoReL ---
        metrics = super().learn(joint_action=joint_action, probs=probs, rewards=returns_modified)
        # print(f"{self.timestep=}")
        # print(f"{self.iteration=}")
        # print(f"{self.joint_policy_mu[0][0]=}")
        # print(f"{self.joint_policy_pi[0][0]=}")
        # sleep(1)
        # print()
        
        # --- At the end of the iteration, update mu and restart the FoReL algo (but keep the pi policy) ---
        if self.timestep == self.n_timesteps_per_iterations:
            # self.joint_policy_mu = self.joint_policy_pi.copy()
            self.iteration += 1
            super().initialize_algorithm(
                game=self.game, 
                joint_policy_pi=self.joint_policy_pi, 
                )

        metrics.update({f"mu_0(a={a})" : self.joint_policy_mu[0][a] for a in range(self.n_actions)})
        return metrics
    
    
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
                
        for i in range(n_players):
            pi_i_a = pi[i][chosen_actions[i]]
            pi_minus_i_a = np.prod([pi[j][chosen_actions[j]] for j in range(n_players) if j != i])
            mu_i_a = mu[i][chosen_actions[i]]
            mu_minus_i_a = np.prod([mu[j][chosen_actions[j]] for j in range(n_players) if j != i])
            returns[i] = returns[i] - eta * np.log(pi_i_a / mu_i_a) + eta * np.log(pi_minus_i_a / mu_minus_i_a)
        
        
        # print(f'{pi[0]=}')
        # print(f'{mu[0]=}')
        # print(f'{pi[1]=}')
        # print(f'{mu[1]=}')
        # print(f'{eta=}')
        # print(f'{chosen_actions=}')
        # print(f'{returns=}')
        # raise
        return returns
            
