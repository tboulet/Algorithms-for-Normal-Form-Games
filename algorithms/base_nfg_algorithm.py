import random
from typing import List, Tuple, Any, Callable
from abc import ABC, abstractmethod

import numpy as np

from core.typing import JointPolicy, Policy
from games.base_nfg_game import BaseNFGGame


class BaseNFGAlgorithm(ABC):
    """The base class for any model-free Normal-Form Game solver.
    It must be able to interact with a pyspiel game object for finding a good joint policy for the game.
    """
    
    @abstractmethod
    def initialize_algorithm(self,
        game: Any,
        ) -> None:
        """Initializes the algorithm.

        Args:
            game (Any): the game to be solved
        """
        pass
    
    @abstractmethod
    def choose_joint_action(self, 
        ) -> Tuple[List[int], List[float]]:
        """Chooses a joint action for the players.

        Returns:
            List[int]: the actions chosen by the players
            List[float]: the probability with which the actions were chosen
        """
        pass
    
    @abstractmethod
    def learn(self,
        joint_action: List[int],
        probs: List[float],
        rewards: List[float],
        ) -> None:
        """Learns from the experience of playing one episode.

        Args:
            joint_actions (List[int]): the joint actions played
            probs (List[float]): the probability of playing the joint action
            rewards (List[float]): the rewards obtained by the players
        """
    
    @abstractmethod
    def do_stop_learning(self,
        ) -> bool:
        """Returns whether the algorithm should stop learning or not.

        Returns:
            bool: whether the algorithm should stop learning or not
        """
        pass
    
    @abstractmethod
    def get_inference_policies(self,
        ) -> JointPolicy:
        """Returns the joint policies used for inference, for evaluation purposes.
        
        Returns:
            JointPolicy: the joint policies used for inference (evaluation)
        """
        pass
    
    
    # Helper methods
    
    def initialize_randomly_joint_policy(self, 
            n_players : int,
            n_actions : int,
            ) -> JointPolicy:
        """Initializes a joint policy randomly.

        Args:
            n_players (int): the number of players
            n_actions (int): the number of actions

        Returns:
            Policy: the initialized joint policy
        """
        joint_policy = np.random.rand(n_players, n_actions)
        joint_policy = joint_policy / np.sum(joint_policy, axis=1, keepdims=True)
        return joint_policy
    
    def get_softmax_joint_policy_from_logits(self,
        joint_logits : List[List[float]],
        ) -> JointPolicy:
        """Define a joint policy from logits, using the softmax function.

        Args:
            joint_logits (List[List[float]]): the logits for each player

        Returns:
            JointPolicy: the joint policy (softmax of the logits)
        """
        return np.array([self.get_softmax_policy_from_logits(logits) for logits in joint_logits])
    
    def get_softmax_policy_from_logits(self,
        logits : List[float],
        ) -> Policy:
        """Define a policy from logits, using the softmax function.

        Args:
            logits (List[float]): the logits

        Returns:
            Policy: the policy (softmax of the logits)
        """
        policy = np.exp(logits)
        policy = policy / policy.sum()
        return policy
    
    def get_uniform_joint_policy(self,
        n_players : int,
        n_actions : int,
        ) -> JointPolicy:
        return np.ones((n_players, n_actions)) / n_actions
    
    def sample_joint_action_probs_from_policy(self,
                                               joint_policy : JointPolicy,
        ) -> Tuple[List[int], List[float]]:
        """Samples a joint action from a joint policy.

        Args:
            joint_policy (JointPolicy): the joint policy

        Returns:
            Tuple[List[int], List[float]]: the joint action and the probability of playing that joint action
        """
        joint_action = []
        joint_action_probs = []
        for i in range(len(joint_policy)):
            if len(joint_policy[i]) == 2:
                action = int(random.random() < joint_policy[i][0])
            else:
                action = random.choices(range(len(joint_policy[i])), weights=joint_policy[i])[0]
            joint_action.append(action)
            joint_action_probs.append(joint_policy[i][action])
        return joint_action, joint_action_probs
    
    
    def get_model_based_q_value(self, 
            game : BaseNFGGame,
            player : int,
            action : int,
            joint_policy : JointPolicy,
            ) -> float:
        """Computes the Q value of a player playing a certain action, using the game object, in a model-based way.

        Args:
            player (int): the player for which we want the Q value
            game (BaseNFGGame): the game object
            action (int): the action for which we want the Q value
            joint_policy (JointPolicy): the joint policy of the players

        Returns:
            float: the Q value of the player playing the action in the joint policy
        """
        assert game.num_players() == 2 and game.num_distinct_actions() == 2, "This method is only implemented for 2-player 2-action games yet"
        q_value = 0
        for b in range(game.num_distinct_actions()):
            joint_action = [action, b] if player == 0 else [b, action]
            q_value += game.get_rewards(joint_action)[player] * joint_policy[1-player][b]
        return q_value        
        
    def is_similar_enough(self,
        joint_policy1: JointPolicy,
        joint_policy2: JointPolicy,
        threshold: float,
    ) -> bool:
        """Checks whether two joint policies are similar enough.

        Args:
            policy1 (JointPolicy): the first policy
            policy2 (JointPolicy): the second policy
            threshold (float): the threshold for the similarity check

        Returns:
            bool: True if the policies are similar enough, False otherwise
        """
        # Implement the similarity check here
        n_players = len(joint_policy1)
        n_actions = len(joint_policy1[0])
        
        for i in range(n_players):
            for a in range(n_actions):
                if abs(joint_policy1[i][a] - joint_policy2[i][a]) > threshold:
                    return False
        return True