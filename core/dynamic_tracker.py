import os
from algorithms.base_nfg_algorithm import BaseNFGAlgorithm
import matplotlib.pyplot as plt


class DynamicTracker:
    def __init__(
        self,
        title: str,
        algo: BaseNFGAlgorithm,
        do_plot_online: bool = True,
        frequency_plot: int = 10000,
        frequency_points_plot: int = 100,
    ):
        """Create a dynamic tracker object that will be used to track the dynamics of the policies during the learning process.

        Args:
            title (str): the name of the run
            algo (BaseNFGAlgorithm): the algorithm used to learn the game
            do_plot_online (bool, optional): Whether to plot. Defaults to True.
            frequency_plot (int, optional): The frequency at which plot is updated with the 'points to plot'. Defaults to 10000.
            frequency_points_plot (int, optional): The frequency at which a point will be added in the 'points to plot'. Defaults to 100.
        """
        self.title = title
        self.algo = algo
        self.do_plot_online = do_plot_online
        self.frequency_plot = frequency_plot
        self.frequency_points_plot = frequency_points_plot
        self.idx_episode_training = 0
        self.list_x = []
        self.list_y = []

        fig, self.ax = plt.subplots()
        (self.previous_positions,) = self.ax.plot(
            [], [], linestyle="-", color="b", label="Trajectory"
        )
        self.current_position = self.ax.scatter(
            [], [], color="r", label="Current position"
        )

        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("probability of taking first action for player 0")
        self.ax.set_ylabel("probability of taking first action for player 1")
        self.ax.set_title("Dynamics of the policies")
        self.ax.legend()
        plt.title(f"Policies Dynamics\n {title}")

    def update(self):
        # Keep track of the policies
        if self.idx_episode_training % self.frequency_points_plot == 0:
            joint_policies_inference = self.algo.get_inference_policies()
            self.list_x.append(joint_policies_inference[0][0])
            self.list_y.append(joint_policies_inference[1][0])

        # Update the plot
        if self.do_plot_online and (
            self.idx_episode_training % self.frequency_plot == 0
        ):
            self.previous_positions.set_data(self.list_x, self.list_y)
            self.current_position.set_offsets(
                [joint_policies_inference[0][0], joint_policies_inference[1][0]]
            )
            plt.pause(0.01)

        # Increment the index of the episode
        self.idx_episode_training += 1

    def show(self):
        plt.show()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.update()
        plt.savefig(path)
        plt.close()
