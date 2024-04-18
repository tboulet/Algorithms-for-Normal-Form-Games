# ML libraries
import random
import numpy as np

# Utils
import datetime
from typing import Any, List, Callable, Optional, Tuple
from matplotlib import pyplot as plt

# Logging
import wandb
from tensorboardX import SummaryWriter
from tqdm import tqdm
import cProfile, pstats

# Config system
import hydra
from omegaconf import DictConfig, OmegaConf
from algorithms.base_nfg_algorithm import BaseNFGAlgorithm

# Project imports
from core.save_and_load import save_joint_policy
from core.utils import to_numeric, try_get, try_get_seed
from core.typing import Policy, JointPolicy
from core.online_plotter import DataPolicyToPlot, get_plotter
from core.nash import compute_nash_conv, compute_nash_equilibrium
from algorithms import algo_name_to_nfg_solver
from games import game_name_to_nfg_solver
from games.base_interactive_nfg_game import InteractiveNFGGame
from games.kuhn_poker import KuhnPokerNFG


@hydra.main(config_path="configs", config_name="play_game.yaml")
def main(config: DictConfig):
    print("Configuration used :")
    print(OmegaConf.to_yaml(config), "\n")
    config = OmegaConf.to_container(config, resolve=True)

    # Get the config parameters
    n_games: int = config["n_games"]
    player_paths: List[str] = config["player_paths"]
    human_position: Optional[int] = config["human_position"]
    seed = try_get_seed(config)
    print(config)

    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)
    BaseNFGAlgorithm.RANDOM_GENERATOR = np.random.default_rng(seed)

    # Get the game
    game_name = config["game"]["game_name"]
    game_config = config["game"]["game_config"]
    GameClass = game_name_to_nfg_solver[game_name]
    game: InteractiveNFGGame = GameClass(**game_config)
    n_players = game.num_players()
    n_actions = game.num_distinct_actions()

    # Load the policies
    joint_policy = []
    for path in player_paths:
        if path == "human":
            joint_policy.append("human")
        else:
            joint_policy.append(np.load(path))

    # Play the games
    average_rewards = []
    for i in range(n_games):
        print(f"\nGame {i+1}/{n_games}")
        try:
            rewards = game.play_game(
                joint_policy_or_human_string=joint_policy,
                human_position=human_position,
            )
            print(f"Gains: {rewards}")
            average_rewards.append(rewards)
        except Exception as e:
            print(f"Error in game {i+1}: {e}")        
    
    print(f"Games played. Average rewards: {np.mean(average_rewards, axis=0)}")

if __name__ == "__main__":
    main()
