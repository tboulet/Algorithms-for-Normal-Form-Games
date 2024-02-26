# ML libraries
import random
import numpy as np

# Utils
import datetime
from typing import Any, List, Callable, Tuple
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
from core.utils import to_numeric, try_get, try_get_seed
from core.typing import Policy, JointPolicy
from core.online_plotter import DataPolicyToPlot, get_plotter
from core.nash import compute_nash_conv, compute_nash_equilibrium
from algorithms import algo_name_to_nfg_solver
from games import game_name_to_nfg_solver


@hydra.main(config_path="configs", config_name="default_config.yaml")
def main(config: DictConfig):
    print("Configuration used :")
    print(OmegaConf.to_yaml(config), "\n")
    config = OmegaConf.to_container(config, resolve=True)

    # Get the config parameters
    n_episodes_training = to_numeric(config["n_episodes_training"])
    nash_computation_method = config["nash_computation_method"]
    seed = try_get_seed(config)

    frequency_metric = to_numeric(config["frequency_metric"])
    do_cli = config["do_cli"]
    frequency_cli = to_numeric(config["frequency_cli"])
    do_tb = config["do_tb"]
    do_wandb = config["do_wandb"]
    wandb_config = config["wandb_config"]
    plot_config = config["plot_config"]
    tqdm_bar = config["tqdm_bar"]

    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)
    BaseNFGAlgorithm.RANDOM_GENERATOR = np.random.default_rng(seed)

    # Get the game
    game_name = config["game"]["game_name"]
    game_config = config["game"]["game_config"]
    GameClass = game_name_to_nfg_solver[game_name]
    game = GameClass(**game_config)
    n_players=game.num_players()
    n_actions=game.num_distinct_actions()
    
    # Get the initial joint policy if specified from the game config
    initial_joint_policy = try_get(config["game"], "initial_joint_policy", None)
    if initial_joint_policy is not None:
        assert len(initial_joint_policy) == n_players and all(
            len(initial_joint_policy[i]) == n_actions[i] for i in range(n_players)
        ), "The initial joint policy is not at the right format"
    
    # Initialize the algorithm for learning in that game
    algo_name = config["algo"]["algo_name"]
    algo_config = config["algo"]["algo_config"]
    AlgoClass = algo_name_to_nfg_solver[algo_name]
    algo = AlgoClass(**algo_config)
    algo.initialize_algorithm(game, joint_policy_pi=initial_joint_policy)

    # Intialize the logging
    run_name = f"[{algo_name}]_[{game_name}]_{datetime.datetime.now().strftime('%dth%mmo_%Hh%Mmin%Ss')}_seed{seed}"
    print(f"Starting run {run_name}")
    plotter = get_plotter(
        n_players=n_players,
        n_actions=n_actions,
        plot_config={"title": f"Policies Dynamics\n {run_name}", **plot_config},
    )
    ne_joint_policy = compute_nash_equilibrium(game, method=nash_computation_method)
    if ne_joint_policy is not None:
        plotter.add_data_policy_to_plot(
            DataPolicyToPlot(
                name="NE",
                joint_policy=ne_joint_policy,
                color="orange",
                marker="x",
            )
        )

    if do_tb:
        writer = SummaryWriter(log_dir=f"tensorboard/{run_name}")
    if do_wandb:
        wandb_run = wandb.init(
            name=run_name,
            config=config,
            **wandb_config,
        )

    for idx_episode_training in tqdm(range(n_episodes_training), disable=not tqdm_bar):

        # Update the dynamic tracker (for visualization of the policies dynamics)
        joint_policy_pi = algo.get_inference_policies()
        plotter.add_data_policy_to_plot(
            DataPolicyToPlot(
                name="previous trajectory",
                joint_policy=joint_policy_pi,
                color="b",
                marker="-",
            )
        )
        plotter.add_data_policy_to_plot(
            DataPolicyToPlot(
                name="current trajectory",
                joint_policy=joint_policy_pi,
                color="r",
                marker="o",
                is_unique=True,
            )
        )
        plotter.update_plot()

        # Choose a joint action
        joint_action, probs = algo.choose_joint_action()

        # Play the joint action and get the rewards
        rewards = game.get_rewards(joint_action).copy()

        # Learn from the experience
        objects_to_log = algo.learn(
            joint_action=joint_action,
            probs=probs,
            rewards=rewards,
        )

        # Log the objects returned by the learn method
        if (
            isinstance(objects_to_log, dict)
            and idx_episode_training % frequency_metric == 0
        ):
            metrics_to_log = {
                k: v for k, v in objects_to_log.items() if isinstance(v, (int, float))
            }
            joint_policies_to_plot = {
                k: v
                for k, v in objects_to_log.items()
                if isinstance(v, DataPolicyToPlot)
            }
            # Log other metrics (like Nash conv)
            metrics_to_log["nash_conv"] = compute_nash_conv(
                game, algo.get_inference_policies()
            )

            # Log the metrics
            if do_tb:
                for metric_name, metric_value in metrics_to_log.items():
                    writer.add_scalar(
                        metric_name, metric_value, global_step=idx_episode_training
                    )
            if do_wandb:
                wandb.log(metrics_to_log, step=idx_episode_training)
            if do_cli and idx_episode_training % frequency_cli == 0:
                print(f"Episode {idx_episode_training} : \n{metrics_to_log}")
            # Log the data policies to plot
            for object_name, data_policy_to_plot in joint_policies_to_plot.items():
                plotter.add_data_policy_to_plot(data_policy_to_plot)

    # At the end of the run, show and save the plot of the dynamics
    plotter.update_plot(force_update=True)
    plotter.save(path=f"logs/{run_name}/dynamics.png")
    plotter.show()

    # Close the logging
    if do_tb:
        writer.close()
    if do_wandb:
        wandb_run.finish()


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats("logs/profile_stats.prof")
