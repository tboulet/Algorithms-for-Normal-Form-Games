from dataclasses import dataclass
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from core.typing import JointPolicy
from core.utils import get_shape


@dataclass
class DataPolicyToPlot:
    """An object representing a joint policy to plot.
    It contains the information of the joint policy, the color and the marker of the data representation, and whether the data policy is unique.

    Args:
        name (str): the name of the metric it represents
        joint_policy (JointPolicy): the joint policy (a list of Policy (an array of n probabilities) for each player)
        color (str): the color of the
        marker (str): the marker of the data representation
        is_unique (bool, optional): whether the data is unique, and should replace all previous data with the same name. Defaults to False.
    """

    name: str
    joint_policy: List[np.ndarray]
    color: str
    marker: str
    is_unique: bool = False


class OnlinePlotter:
    def __init__(
        self,
        title: str,
        do_plot_online: bool = True,
        update_frequency: int = 10000,
        pause_time: float = 0.01,
        do_plot_final: bool = True,
    ):
        """Create an object to generate a 2D plot online and visualize it.

        Args:
            title (str): the title of the plot
            do_plot_online (bool, optional): Whether to plot. Defaults to True.
            update_frequency (int, optional): The frequency at which plot is updated with the 'points to plot'. Defaults to 10000.
            pause_time (float, optional): The pause time between each update. Defaults to 0.01.
            do_plot_final (bool, optional): Whether to plot the final plot. Defaults to True.
        """
        self.title = title
        self.do_plot_online = do_plot_online
        self.update_frequency = update_frequency
        self.pause_time = pause_time
        self.do_plot_final = do_plot_final
        
        # Create plot memory objects
        self.name_dataPolicy_to_list_x_list_y: Dict[
            str, Tuple[List[float], List[float]]
        ] = {}
        self.name_dataPolicy_to_line2d: Dict[str, plt.Line2D] = {}
        self.plot_timestep = 0

        # Create the plot
        _, self.ax = plt.subplots()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.legend()
        plt.title(title)

    def add_data_policy_to_plot(
        self,
        data_policy: DataPolicyToPlot,
    ):
        """Add a data policy to the list of data policies to plot.

        Args:
            data_policy (DataPolicyToPlot): the data policy to add to the plot
        """

        # Create the data policy list and line2d if it does not exist
        if data_policy.name not in self.name_dataPolicy_to_list_x_list_y:
            self.name_dataPolicy_to_list_x_list_y[data_policy.name] = ([], [])
            (self.name_dataPolicy_to_line2d[data_policy.name],) = self.ax.plot(
                [],
                [],
                data_policy.marker,
                color=data_policy.color,
                label=data_policy.name,
            )
            self.ax.legend()
        # If the data policy is unique, remove all previous data policies with the same name
        if data_policy.is_unique:
            self.name_dataPolicy_to_list_x_list_y[data_policy.name] = ([], [])

        if len(get_shape(data_policy.joint_policy)) == 2:
            # If the joint_policy represents a joint policy (so a (n_players, n_action_per_players) shaped object), add the data policy to the list
            self.name_dataPolicy_to_list_x_list_y[data_policy.name][0].append(
                data_policy.joint_policy[0][0]
            )
            self.name_dataPolicy_to_list_x_list_y[data_policy.name][1].append(
                data_policy.joint_policy[1][0]
            )
        elif len(get_shape(data_policy.joint_policy)) == 3:
            # If the joint_policy represents a list of joint policies (so a (n, n_players, n_action_per_players) shaped object), add each joint policy to the list
            for joint_policy_i in data_policy.joint_policy:
                self.name_dataPolicy_to_list_x_list_y[data_policy.name][0].append(
                    joint_policy_i[0][0]
                )
                self.name_dataPolicy_to_list_x_list_y[data_policy.name][1].append(
                    joint_policy_i[1][0]
                )
        else:
            raise ValueError(
                "The joint_policy should be a (n, n_players, n_action_per_players) shaped objects, or a (n_players, n_action_per_players) shaped object."
            )

    def update_plot(self, force_update: bool = False):
        """Update the online plot with the data policies currently in memory.

        Args:
            force_update (bool, optional): whether to force plotting in spite of the update frequency constraints. Defaults to False.
        """
        # Update the plot
        if force_update or (
            self.do_plot_online and (self.plot_timestep % self.update_frequency == 0)
        ):
            for name_dataPolicy, (
                list_x,
                list_y,
            ) in self.name_dataPolicy_to_list_x_list_y.items():
                self.name_dataPolicy_to_line2d[name_dataPolicy].set_data(list_x, list_y)
            plt.pause(0.01)

        # Increment the index of the plot object
        self.plot_timestep += 1

    def show(self):
        """Show the plot."""
        plt.show()

    def save(self, path: str):
        """Save the plot to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.update_plot()
        plt.savefig(path)
        plt.close()

    def try_final_plot(self):
        """Try to plot the final plot."""
        if self.do_plot_final:
            self.update_plot(force_update=True)
            self.show()
            
            
class OnlinePlotterForACertainPlayer(OnlinePlotter):
    def __init__(
        self,
        title: str,
        player_index: int,
        n_actions: int,
        do_plot_online: bool = True,
        update_frequency: int = 10000,
        pause_time: float = 0.01,
        do_plot_final: bool = True,
    ):
        """Create an object to generate a 2D plot online and visualize it.

        Args:
            title (str): the title of the plot
            n_actions (int): the number of actions for the player
            do_plot_online (bool, optional): Whether to plot. Defaults to True.
            update_frequency (int, optional): The frequency at which plot is updated with the 'data policies to plot'. Defaults to 10000.
            pause_time (float, optional): The pause time between each update. Defaults to 0.01.
            do_plot_final (bool, optional): Whether to plot the final plot. Defaults to True.
        """

        # Variables
        self.title = title
        self.do_plot_online = do_plot_online
        self.update_frequency = update_frequency
        self.pause_time = pause_time
        self.do_plot_final = do_plot_final
        self.player_index = player_index
        self.n_actions = n_actions

        if self.n_actions < 3:
            raise ValueError("Number of sides should be at least 3.")

        # Generating the vertices of the polygon
        angles_with_endpoint = np.linspace(0, 2 * np.pi, n_actions + 1, endpoint=True)
        vertices_x_with_endpoint = np.cos(angles_with_endpoint)
        vertices_y_with_endpoint = np.sin(angles_with_endpoint)
        self.angles = angles_with_endpoint[:-1]
        self.vertices_x = np.cos(self.angles)
        self.vertices_y = np.sin(self.angles)

        # Create the plot
        fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.title.set_text(title)
        self.ax.plot(vertices_x_with_endpoint, vertices_y_with_endpoint, marker="o")
        self.ax.axis("equal")  # idk what this does
        for a in range(n_actions):
            self.ax.text(
                self.vertices_x[a],
                self.vertices_y[a],
                str(a),
                fontsize=12,
                ha="right",
                va="bottom",
            )

        # Create plot memory objects
        self.name_dataPolicy_to_list_x_list_y: Dict[str, List[List[float]]] = {}
        self.name_dataPolicy_to_line2d: Dict[str, plt.Line2D] = {}
        self.plot_timestep = 0

    def add_data_policy_to_plot(
        self,
        data_policy: DataPolicyToPlot,
    ):
        # Create the data policy list and line2d if it does not exist
        if data_policy.name not in self.name_dataPolicy_to_list_x_list_y:
            self.name_dataPolicy_to_list_x_list_y[data_policy.name] = ([], [])
            (self.name_dataPolicy_to_line2d[data_policy.name],) = self.ax.plot(
                [],
                [],
                data_policy.marker,
                color=data_policy.color,
                label=data_policy.name,
            )
            self.ax.legend()
        # If the data policy is unique, remove all previous data policies with the same name
        if data_policy.is_unique:
            self.name_dataPolicy_to_list_x_list_y[data_policy.name] = ([], [])
            
        # Add the data policy to the list
        if len(get_shape(data_policy.joint_policy)) == 2:
            # If the joint_policy represents a joint policy (so a (n_players, n_action_per_players) shaped object), add the data policy to the list
            point_x = np.sum(self.vertices_x * data_policy.joint_policy[self.player_index])
            point_y = np.sum(self.vertices_y * data_policy.joint_policy[self.player_index])
            self.name_dataPolicy_to_list_x_list_y[data_policy.name][0].append(point_x)
            self.name_dataPolicy_to_list_x_list_y[data_policy.name][1].append(point_y)
        elif len(get_shape(data_policy.joint_policy)) == 3:
            # If the joint_policy represents a list of joint policies (so a (n, n_players, n_action_per_players) shaped object), add each joint policy to the list
            for joint_policy_i in data_policy.joint_policy:
                point_x = np.sum(self.vertices_x * joint_policy_i[self.player_index])
                point_y = np.sum(self.vertices_y * joint_policy_i[self.player_index])
                self.name_dataPolicy_to_list_x_list_y[data_policy.name][0].append(point_x)
                self.name_dataPolicy_to_list_x_list_y[data_policy.name][1].append(point_y)

class OnlinePlotterForNPlayers(OnlinePlotter):
    """An object to generate a 2D plot online and visualize it for each player. It contains a plotter for each player."""

    def __init__(
        self,
        title: str,
        n_players: int,
        n_actions_by_player: int,
        do_plot_online: bool = True,
        update_frequency: int = 10000,
        pause_time: float = 0.01,
        do_plot_final : bool = True,
    ):
        self.player_idx_to_plotter: Dict[int, OnlinePlotterForACertainPlayer] = {
            i: OnlinePlotterForACertainPlayer(
                title=f"[Player {i}] - {title}",
                player_index=i,
                n_actions=n_actions_by_player[i],
                do_plot_online=do_plot_online,
                update_frequency=update_frequency,
                pause_time=pause_time,
                do_plot_final=do_plot_final,
            )
            for i in range(n_players)
        }
        self.do_plot_final = do_plot_final

    def add_data_policy_to_plot(
        self,
        data_policy: DataPolicyToPlot,
    ):
        if len(get_shape(data_policy.joint_policy)) == 2:
            # If the joint_policy represents a joint policy (so a (n_players, n_action_per_players) shaped object), the number of players is the length of the joint policy
            n_players = len(data_policy.joint_policy)
        elif len(get_shape(data_policy.joint_policy)) == 3:
            # If the joint_policy represents a list of joint policies (so a (n, n_players, n_action_per_players) shaped object), the number of players is the length of the first joint policy
            n_players = len(data_policy.joint_policy[0])
        else:
            raise ValueError(
                "The joint_policy should be a (n, n_players, n_action_per_players) shaped objects, or a (n_players, n_action_per_players) shaped object."
            )

        for i in range(n_players):
            self.player_idx_to_plotter[i].add_data_policy_to_plot(data_policy)

    def update_plot(self, force_update: bool = False):
        for plotter in self.player_idx_to_plotter.values():
            plotter.update_plot(force_update=force_update)

    def try_final_plot(self):
        if self.do_plot_final:
            for plotter in self.player_idx_to_plotter.values():
                plotter.update_plot(force_update=True)
            plotter.show()
            
def get_plotter(
    n_players: int,
    n_actions: List[int],
    plot_config: Dict[str, Any],
) -> OnlinePlotter:
    """Get a plotter object from a configuration.

    Args:
        n_players (int): the number of players
        n_actions (List[int]): the number of actions for each player
        plot_config (Dict[str, Any]): the configuration of the plot

    Returns:
        OnlinePlotter: the plotter object
    """
    assert n_players > 1, "This function is only for n_players > 1"
    assert len(n_actions) == n_players, "n_actions should have n_players elements"
    assert all(
        [n > 1 for n in n_actions]
    ), "All elements of n_actions should be greater than 1"

    # If each player has a different number of actions, use a plotter for each player
    if np.unique(n_actions).shape[0] != 1:
        return OnlinePlotterForNPlayers(
            **plot_config, n_players=n_players, n_actions_by_player=n_actions
        )
    # In the 2-player 2-action case, use the regular plotter
    elif n_players == 2 and n_actions[0] == 2:
        return OnlinePlotter(**plot_config)
    # If there are more than 2 players, or more than 2 actions, use a plotter for each player
    else:
        return OnlinePlotterForNPlayers(
            **plot_config, n_players=n_players, n_actions_by_player=n_actions
        )
