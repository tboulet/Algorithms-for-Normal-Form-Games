from typing import List, Tuple, Any, Callable
from abc import ABC, abstractmethod

from core.typing import JointPolicy, Policy


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