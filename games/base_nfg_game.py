from abc import ABC, abstractmethod
from typing import List

from core.typing import JointPolicy, Policy


class BaseNFGGame(ABC):
    """The base class for any Normal Form Game"""    
    
    @abstractmethod
    def get_rewards(self, joint_action : List[int]) -> List[float]:
        """Return the rewards for each player"""
        pass
    
    @abstractmethod
    def num_distinct_actions(self) -> int:
        """Return the number of distinct actions"""
        pass
    
    @abstractmethod
    def num_players(self) -> int:
        """Return the number of players"""
        pass