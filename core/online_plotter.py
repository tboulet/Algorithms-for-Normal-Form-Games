from dataclasses import dataclass
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any


@dataclass
class PointToPlot:
    """An object representing a point in 2D space to be plotted.
    It contains the information of the coordinates but also the color and the marker of the point.

    Args:
        name (str): the name of the metric it represents
        coords (Tuple[float, float]): the coordinates of the point
        color (str): the color of the point
        marker (str): the marker of the point
        is_unique (bool, optional): whether the point is unique and should replace any previous point with the same name. Defaults to False.
    """

    name: str
    coords: Tuple[float, float]
    color: str
    marker: str
    is_unique: bool = False


class OnlinePlotter:
    def __init__(
        self,
        title: str,
        x_label: str,
        y_label: str,
        do_plot_online: bool = True,
        update_frequency: int = 10000,
        pause_time: float = 0.01,
    ):
        """Create an object to generate a 2D plot online and visualize it.

        Args:
            title (str): the title of the plot
            x_label (str): the label of the x axis
            y_label (str): the label of the y axis
            do_plot_online (bool, optional): Whether to plot. Defaults to True.
            update_frequency (int, optional): The frequency at which plot is updated with the 'points to plot'. Defaults to 10000.
            pause_time (float, optional): The pause time between each update. Defaults to 0.01.
        """
        self.title = title
        self.do_plot_online = do_plot_online
        self.update_frequency = update_frequency
        self.pause_time = pause_time

        # Create plot memory objects
        self.point_name_to_list_x_list_y: Dict[str, Tuple[List[float], List[float]]] = (
            {}
        )
        self.point_names_to_line2d: Dict[str, plt.Line2D] = {}
        self.plot_timestep = 0

        # Create the plot
        _, self.ax = plt.subplots()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.legend()
        plt.title(title)

    def add_point(
        self,
        point: PointToPlot,
    ):
        """Add a point to the list of points to plot.

        Args:
            point (PointToPlot): the point to add to the plot
        """

        # Create the point list and line2d if it does not exist
        if point.name not in self.point_name_to_list_x_list_y:
            self.point_name_to_list_x_list_y[point.name] = ([], [])
            (self.point_names_to_line2d[point.name],) = self.ax.plot(
                [], [], point.marker, color=point.color, label=point.name
            )
            self.ax.legend()
        # If the point is unique, remove all previous points with the same name
        if point.is_unique:
            self.point_name_to_list_x_list_y[point.name] = ([], [])
        # Add the point to the list
        self.point_name_to_list_x_list_y[point.name][0].append(point.coords[0])
        self.point_name_to_list_x_list_y[point.name][1].append(point.coords[1])

    def update_plot(self, force_update: bool = False):
        """Update the online plot with the points currently in memory.

        Args:
            force_update (bool, optional): whether to force plotting in spite of the update frequency constraints. Defaults to False.
        """
        # Update the plot
        if force_update or (
            self.do_plot_online and (self.plot_timestep % self.update_frequency == 0)
        ):
            for point_name, (
                list_x,
                list_y,
            ) in self.point_name_to_list_x_list_y.items():
                self.point_names_to_line2d[point_name].set_data(list_x, list_y)
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
