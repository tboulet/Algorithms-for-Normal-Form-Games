import random
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod

import numpy as np
from core.online_plotter import DataPolicyToPlot

from core.typing import JointPolicy, Policy
from games.base_nfg_game import BaseNFGGame


class BaseNFGAlgorithm(ABC):
    """The base class for any model-free Normal-Form Game solver.
    It must be able to interact with a pyspiel game object for finding a good joint policy for the game.
    """

    RANDOM_GENERATOR = np.random.default_rng(42)

    @abstractmethod
    def initialize_algorithm(
        self,
        game: Any,
        joint_policy_pi: Optional[JointPolicy] = None,
    ) -> None:
        """Initializes the algorithm.

        Args:
            game (Any): the game to be solved
            joint_policy_pi (Optional[JointPolicy], optional): if not None, force the algorithm to use this joint policy (if possible).
        """
        pass

    @abstractmethod
    def choose_joint_action(
        self,
    ) -> Tuple[List[int], List[float]]:
        """Chooses a joint action for the players.

        Returns:
            List[int]: the actions chosen by the players
            List[float]: the probability with which the actions were chosen
        """
        pass

    @abstractmethod
    def learn(
        self,
        joint_action: List[int],
        probs: List[float],
        rewards: List[float],
    ) -> Optional[Dict[str, Union[float, DataPolicyToPlot]]]:
        """Learns from the experience of playing one episode.

        Args:
            joint_actions (List[int]): the joint actions played
            probs (List[float]): the probability of playing the joint action
            rewards (List[float]): the rewards obtained by the players

        Returns:
            Optional[Dict[str, Union[float, DataPolicyToPlot]]]: the objects to log, as a dictionnary with
            the name of the object as key and either a numerical metric of a dataPolicy to plot as value.
        """

    @abstractmethod
    def get_inference_policies(
        self,
    ) -> JointPolicy:
        """Returns the joint policies used for inference, for evaluation purposes.

        Returns:
            JointPolicy: the joint policies used for inference (evaluation)
        """
        pass

    # Helper methods

    def action_index_to_action_repr(
        self,
        player: int,
        action: int,
    ) -> str:
        """Returns the string representation of an action.

        Args:
            player (int): the player
            action (int): the action

        Returns:
            str: the string representation of the action-th action of the player-th player
        """
        return action

    def initialize_randomly_joint_policy(
        self,
        n_actions: List[int],
    ) -> JointPolicy:
        """Initializes a joint policy randomly.

        Args:
            n_actions (List[int]): the number of actions for each player

        Returns:
            JointPolicy: the initialized joint policy
        """
        joint_policy = [
            self.RANDOM_GENERATOR.random(n_player_actions)
            for n_player_actions in n_actions
        ]

        for i in range(len(joint_policy)):
            joint_policy[i] = joint_policy[i] / joint_policy[i].sum()

        return joint_policy

    def get_softmax_joint_policy_from_logits(
        self,
        joint_logits: List[np.ndarray],
    ) -> JointPolicy:
        """Define a joint policy from logits, using the softmax function.

        Args:
            joint_logits (List[np.ndarray]): the logits for each player

        Returns:
            JointPolicy: the joint policy (softmax of the logits)
        """
        return np.array(
            [self.get_softmax_policy_from_logits(logits) for logits in joint_logits]
        )

    def get_softmax_policy_from_logits(
        self,
        logits: np.ndarray,
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

    def get_uniform_joint_policy(self, n_actions: List[int]) -> JointPolicy:
        return [np.ones((n_action,)) / n_action for n_action in n_actions]

    def sample_joint_action_probs_from_policy(
        self,
        joint_policy: JointPolicy,
    ) -> Tuple[List[int], List[float]]:
        """Samples a joint action from a joint policy.

        Args:
            joint_policy (JointPolicy): the joint policy

        Returns:
            Tuple[List[int], List[float]]: the joint action and the probability of playing that joint action
        """
        joint_action = [0] * len(joint_policy)
        joint_action_probs = [0.0] * len(joint_policy)
        for i in range(len(joint_policy)):
            if len(joint_policy[i]) == 2:
                action = int(random.random() > joint_policy[i][0])
            else:
                action = self.RANDOM_GENERATOR.choice(
                    len(joint_policy[i]), p=joint_policy[i]
                )
            joint_action[i] = action
            joint_action_probs[i] = joint_policy[i][action]
        return joint_action, joint_action_probs

    def is_similar_enough(
        self,
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
        return not any(
            np.any(np.abs(joint_policy1[i] - joint_policy2[i]) > threshold)
            for i in range(len(joint_policy1))
        )
