# Imports
from turtle import color

import numpy as np
from core.utils import to_numeric

import datetime
from typing import Any, List, Callable, Tuple
from matplotlib import pyplot as plt
from tqdm import tqdm

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
    
    # Create a plot
    joint_policies_inference = algo.get_inference_policies()
    xy = [joint_policies_inference[0][0], joint_policies_inference[1][0]]
    xy_history = [xy]
    fig, ax = plt.subplots()
    current_position = ax.scatter(*xy, c='r', label='Current Position')
    previous_positions, = ax.plot([], [], linestyle='-', color='b', label='Trajectory')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('probability of taking first action for player 0')
    ax.set_ylabel('probability of taking first action for player 1')
    ax.set_title('Dynamics of the policies')
    ax.legend()
    plt.title('Policies')
    
    for idx_episode_training in range(n_episodes_training):
        
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
        
        # Update the policies plot
        joint_policies_inference = algo.get_inference_policies()
        xy = [joint_policies_inference[0][0], joint_policies_inference[1][0]]
        current_position.set_offsets(xy)
        xy_history.append(xy)
        xy_history_array = np.array(xy_history)
        previous_positions.set_data(xy_history_array[:, 0], xy_history_array[:, 1])
        plt.pause(0.001)
        
        # Check if we should stop learning
        if algo.do_stop_learning():
            break
        

if __name__ == "__main__":
    main()