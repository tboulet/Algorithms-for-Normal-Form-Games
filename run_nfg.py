# Imports

import numpy as np
from core.utils import to_numeric

import datetime
from typing import Any, List, Callable, Tuple
from matplotlib import pyplot as plt
from tqdm import tqdm
import cProfile, pstats

import hydra
from omegaconf import DictConfig, OmegaConf

# Define some types for lisibility
Policy = List[float]
JointPolicy = List[Policy]   # if p is of type Policy, then p[i][a] = p_i(a)
Game = Any

from algorithms import algo_name_to_nfg_solver
from games import game_name_to_nfg_solver

@hydra.main(config_path="configs", config_name="default_config.yaml")
def main(config: DictConfig):
    print("Configuration used :")
    print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_container(config, resolve=True)
    
    # Get the config parameters
    n_episodes_training = to_numeric(config["n_episodes_training"])
    do_plot_online = config["do_plot_online"]
    frequency_plot = to_numeric(config["frequency_plot"])
    tqdm_bar = config["tqdm_bar"]
    
    # Get the game
    game_name = config["game"]["game_name"]
    game_config = config["game"]["game_config"]
    GameClass = game_name_to_nfg_solver[game_name]
    game = GameClass(**game_config)
    
    # Initialize the algorithm for learning in that game
    algo_name = config["algo"]["algo_name"]
    algo_config = config["algo"]["algo_config"]
    AlgoClass = algo_name_to_nfg_solver[algo_name]
    algo = AlgoClass(**algo_config)
    algo.initialize_algorithm(game)
    
    # Initialize policies tracker
    joint_policies_inference = algo.get_inference_policies()
    list_x = [joint_policies_inference[0][0]]
    list_y = [joint_policies_inference[1][0]]
    fig, ax = plt.subplots()
    previous_positions, = ax.plot([], [], linestyle='-', color='b', label='Trajectory')
    current_position = ax.scatter(list_x, list_y, color='r', label='Current position')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('probability of taking first action for player 0')
    ax.set_ylabel('probability of taking first action for player 1')
    ax.set_title('Dynamics of the policies')
    ax.legend()
    plt.title('Policies')
    
    for idx_episode_training in tqdm(range(n_episodes_training), disable=not tqdm_bar):
        
        # Choose a joint action
        joint_action, probs = algo.choose_joint_action()
        
        # Play the joint action and get the rewards
        rewards = game.get_rewards(joint_action)
        
        # Learn from the experience
        algo.learn(
            joint_action=joint_action, 
            probs=probs, 
            rewards=rewards,
            )
        
        # Keep track of the policies
        joint_policies_inference = algo.get_inference_policies()
        list_x.append(joint_policies_inference[0][0])
        list_y.append(joint_policies_inference[1][0])
        if do_plot_online or idx_episode_training % frequency_plot == 0:
            previous_positions.set_data(list_x, list_y)
            current_position.set_offsets([list_x[-1], list_y[-1]])
            plt.pause(0.1)
    
    
    # Plot the policies
    if not do_plot_online:
        previous_positions.set_data(list_x, list_y)
        current_position.set_offsets([list_x[-1], list_y[-1]])
        plt.show()

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats('logs/profile.prof')