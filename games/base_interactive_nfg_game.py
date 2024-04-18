# Interface for the Interactivee NFG games, which are games implements
# a method for playing the game against a human player.

# ML libraries
import random
import numpy as np

# Utils
import datetime
from typing import Any, List, Callable, Optional, Tuple
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod

# Logging
import wandb
from tensorboardX import SummaryWriter
from tqdm import tqdm
import cProfile, pstats

# Config system
import hydra
from omegaconf import DictConfig, OmegaConf

# Project imports
from core.typing import Policy, JointPolicy
from games.base_nfg_game import BaseNFGGame


class InteractiveNFGGame(BaseNFGGame):
    @abstractmethod
    def play_game(
        self,
        joint_policy_or_human_string: JointPolicy,
        human_position: Optional[int],
    ) -> List[float]:
        """Play a game in an interactive way, possibly against human player(s)
        
        Args:
            joint_policy_or_human_string (JointPolicy): a list of policy or 'human' for human player(s)
            human_position (Optional[int]): the position of a forced human player
            
        Returns:
            List[float]: the payoffs of the game
        """
        
