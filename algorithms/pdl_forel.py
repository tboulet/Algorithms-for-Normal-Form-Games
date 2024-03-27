import sys
import numpy as np
from typing import Any, Dict, List, Optional

from algorithms.forel import Forel
from core.online_plotter import DataPolicyToPlot
from core.scheduler import Scheduler
from core.utils import to_numeric
from games.base_nfg_game import BaseNFGGame
from core.typing import JointPolicy
from copy import deepcopy
import random
import statistics

class PDLForel(Forel):
    def __init__(
        self,
        # FoReL specific parameters
        forel_config: Dict[str, Any],
        # Iterated FoReL specific parameters
        n_timesteps_per_iterations: int,
        do_mu_update: bool,
        do_linear_interpolation_mu: bool,
        eta_scheduler_config: dict,
        n_policies_to_sample: int,
        method_avg: str,
        alternate_lyap_pc: str
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
            do_mu_update (bool): whether to update the mu policy at each iteration, or not (in this case the mu used is the initial pi policy)
            do_linear_interpolation_mu (bool): whether to use a linear interpolation between mu_{k-1} and mu_k, or not (in this case mu is brutally updated at the end of each iteration)
            eta_scheduler_config (dict): the configuration of the eta scheduler. It should contain the following keys:
                - type (str): the type of scheduler (either "constant" or "linear")
                - start_value (float): the initial value of eta
                - end_value (float): the final value of eta
                - n_iterations (int): the number of iterations over which eta will decrease
            n_policies_to_sample (int) : number of policies to sample for the averaging operation
            method_avg (str) : can be 'arithmetic' or 'geometric', it is the type of average we use
            alternate_lyap_pc (str): indicates wether we follow an alternate lyapunov - Forel or basic lyapunov method
        """
        super().__init__(**forel_config)
        self.n_timesteps_per_iterations = to_numeric(n_timesteps_per_iterations)
        self.do_mu_update = do_mu_update
        self.do_linear_interpolation_mu = do_linear_interpolation_mu
        self.alternate_lyap_pc = alternate_lyap_pc
        self.eta_scheduler = Scheduler(**eta_scheduler_config)
        #self.eta_scheduler.set_type("exponential")
        self.n_policies_to_sample = n_policies_to_sample
        self.list_policies = []
        self.nb_last_policies = 50000
        # update pour l'integrer dans la config
        self.avg = method_avg
        self.lyapunov = True
        # ce qu'on veut faire c'est avoir un eta qui decay tres vite,
        # qu'on puisse aboutir à une situation de recurrence
        # qu'on puisse sampler nos politiques
        # update le mu accordingly
        # et ensuite reinitialiser le eta et continuer

    # Interface methods

    def get_list_policies(self):
        return self.list_policies
    
    def add_past_policy(self,policy):
        self.list_policies.append(policy)

    def set_joint_policy_pi(self,policy):
        self.joint_policy_pi = policy
    
    def set_joint_policy_mu(self,policy):
        self.joint_policy_mu = policy
    
    def is_alternate_lyap_pc(self):
        return self.alternate_lyap_pc
    
    def is_lyap(self):
        if self.is_alternate_lyap_pc():

            if self.iteration % 2 == 0:
                return True
            else:
                return False
        else:
            return True

    def initialize_algorithm(
        self,
        game: BaseNFGGame,
        joint_policy_pi: Optional[JointPolicy] = None,
    ) -> None:
        super().initialize_algorithm(game=game, joint_policy_pi=joint_policy_pi)
        self.iteration: int = 0
        self.joint_policy_mu_k_minus_1 = self.initialize_randomly_joint_policy(
            n_actions=self.n_actions
        )
        self.joint_policy_mu = deepcopy(self.joint_policy_pi)

    def get_n_policies_to_sample(self):
        return self.n_policies_to_sample
    
    def sample_policies(self,n_last_policies):
        return random.sample(self.get_list_policies()[-n_last_policies:],self.get_n_policies_to_sample())
    
    def get_average_policies(self,policies):
        if self.avg == "geometric":
            avg_sampled_policy = np.array([statistics.geometric_mean([elt[0][0] for elt in policies]),statistics.geometric_mean([elt[1][0] for elt in policies])])
        else:
            avg_sampled_policy = np.array([statistics.mean([elt[0][0] for elt in policies]),statistics.mean([elt[1][0] for elt in policies])])
        avg_counter_policy = np.array([1,1]) - avg_sampled_policy
        return np.array([[avg_sampled_policy[0],avg_counter_policy[0]],[avg_sampled_policy[1],avg_counter_policy[1]]])

    def learn(
        self,
        joint_action: List[int],
        probs: List[float],
        rewards: List[float],
    ) -> None:

        # --- Modify the rewards ---

        # Compute the linear interpolation rate if needed
        if self.do_mu_update and self.do_linear_interpolation_mu:
            interpolation_rate_alpha = self.get_interpolation_rate_alpha()

        # Add the Lyapunov reward to the rewards
        lyap_reward_from_mu = self.lyapunov_reward(
            chosen_actions=joint_action,
            pi=self.joint_policy_pi,
            mu=self.joint_policy_mu,
        )

        # If we use linear interpolation, we also need to compute the Lyapunov reward from mu_{k-1}
        if self.do_mu_update and self.do_linear_interpolation_mu:
            lyap_reward_from_mu_k_minus_1 = self.lyapunov_reward(
                chosen_actions=joint_action,
                pi=self.joint_policy_pi,
                mu=self.joint_policy_mu_k_minus_1,
            )
            lyap_reward_from_mu = (
                (1 - interpolation_rate_alpha) * lyap_reward_from_mu_k_minus_1
                + interpolation_rate_alpha * lyap_reward_from_mu
            )

        # Add the Lyapunov reward to the rewards
        rewards += lyap_reward_from_mu

        # --- Do one learning step of FoReL ---
        # c'est ici que la politique est update
        metrics = super().learn(joint_action=joint_action, probs=probs, rewards=rewards)
        self.add_past_policy(self.get_inference_policies())

        # --- At the end of the iteration, update mu and restart the FoReL algo (but keep the pi policy) ---
        if self.timestep == self.n_timesteps_per_iterations:
            if self.do_mu_update:
                self.joint_policy_mu_k_minus_1 = deepcopy(self.joint_policy_mu)
                self.joint_policy_mu = deepcopy(self.get_average_policies(self.sample_policies(self.nb_last_policies)))
            self.iteration += 1
            super().initialize_algorithm(
                game=self.game,
                joint_policy_pi=self.joint_policy_pi,
            )

        # Add the metrics and dataPolicies to plot
        metrics["iteration"] = self.iteration
        metrics["timestep"] = self.timestep
        metrics["eta"] = self.get_eta()

        for i in range(self.n_players):
            metrics[f"reward_modif/reward_modif_{i}"] = rewards[i]
            for a in range(self.n_actions[i]):
                metrics[f"mu_{i}/mu_{i}_{a}"] = self.joint_policy_mu[i][a]
        metrics["mu"] = DataPolicyToPlot(
            name="μ",
            joint_policy=self.joint_policy_mu,
            color="g",
            marker="o",
            is_unique=True,
        )
        metrics["mu_k"] = DataPolicyToPlot(
            name="μ_k",
            joint_policy=self.joint_policy_mu_k_minus_1,
            color="g",
            marker="x",
        )
        if self.do_mu_update and self.do_linear_interpolation_mu:
            metrics["interpolation_rate_alpha"] = interpolation_rate_alpha
            metrics["mu_interp"] = DataPolicyToPlot(
                name="mu_interp",
                joint_policy=interpolation_rate_alpha * self.joint_policy_mu
                + (1 - interpolation_rate_alpha) * self.joint_policy_mu_k_minus_1,
                color="c",
                marker="o",
                is_unique=True,
            )
        return metrics

    # Helper methods
    def get_eta(self) -> float:
        if self.is_lyap():
            return self.eta_scheduler.get_value(
                self.iteration * self.n_timesteps_per_iterations*0  + self.timestep
            )
        else:
            return 0

    def get_interpolation_rate_alpha(self) -> float:
        return Scheduler(
            type="linear",
            start_value=0,
            end_value=1,
            n_steps=self.n_timesteps_per_iterations // 2,
            upper_bound=1,
            lower_bound=0,
        ).get_value(self.timestep)

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
            oppenent_term = (
                opponent_policy
                * np.log(opponent_policy / self.joint_policy_mu[1 - player]).sum()
            )
            player_term = np.log(
                self.joint_policy_pi[player] / self.joint_policy_mu[player]
            )

            self.joint_q_values[player] = self.joint_q_values[player] + eta * (
                -player_term + oppenent_term
            )
