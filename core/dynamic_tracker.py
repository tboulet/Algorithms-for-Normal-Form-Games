import os
from algorithms.base_nfg_algorithm import BaseNFGAlgorithm
import matplotlib.pyplot as plt

class DynamicTracker:
    def __init__(self, 
            name : str, 
            algo : BaseNFGAlgorithm,
            do_plot_online : bool,
            ):
        self.name = name
        self.algo = algo
        self.list_x = []
        self.list_y = []
        
        fig, self.ax = plt.subplots()
        self.previous_positions, = self.ax.plot([], [], linestyle='-', color='b', label='Trajectory')
        self.current_position = self.ax.scatter([], [], color='r', label='Current position')
        # self.update(do_update_plot=do_plot_online)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('probability of taking first action for player 0')
        self.ax.set_ylabel('probability of taking first action for player 1')
        self.ax.set_title('Dynamics of the policies')
        self.ax.legend()
        plt.title(f'Policies Dynamics\nRun: {name}')
        
    def update(self, do_update_plot : bool):
        # Keep track of the policies
        joint_policies_inference = self.algo.get_inference_policies()
        self.list_x.append(joint_policies_inference[0][0])
        self.list_y.append(joint_policies_inference[1][0])
        
        # Update the plot
        if do_update_plot:
            self.previous_positions.set_data(self.list_x, self.list_y)
            self.current_position.set_offsets([joint_policies_inference[0][0], joint_policies_inference[1][0]])
            plt.pause(0.01)
    
    def show(self):
        plt.show()
        
    def save(self, path : str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.update(do_update_plot=True)
        plt.savefig(path)
        plt.close()