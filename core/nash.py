import sys
from games.base_nfg_game import BaseNFGGame
from core.typing import JointPolicy, Policy
from typing import List

import numpy as np
from scipy.optimize import linprog
import copy


def get_expected_payoff(game: BaseNFGGame, joint_policy: JointPolicy) -> np.ndarray:
    utility_matrix = game.get_utility_matrix()

    expected_payoff = utility_matrix * joint_policy[0][:, np.newaxis, np.newaxis]
    expected_payoff = expected_payoff * joint_policy[1][np.newaxis, :, np.newaxis]

    return np.sum(expected_payoff, axis=(0, 1))


def get_expected_strategy_payoff(
    game: BaseNFGGame, player: int, action: int, joint_policy: JointPolicy
) -> float:
    joint_policy = copy.deepcopy(joint_policy)
    joint_policy[player] = np.zeros_like(joint_policy[player])
    joint_policy[player][action] = 1.0

    return get_expected_payoff(game, joint_policy)[player]


def get_best_response(
    game: BaseNFGGame, player: int, joint_policy: JointPolicy
) -> tuple[List[int], float]:
    model_based_q_value = game.get_model_based_q_value(player, joint_policy)
    max_utility = np.max(model_based_q_value)

    return np.nonzero(model_based_q_value == max_utility)[0], max_utility


def compute_nash_conv(game: BaseNFGGame, joint_policy: JointPolicy) -> float:
    nash_conv = 0.0
    expected_payoff = get_expected_payoff(game, joint_policy)

    for player in range(game.num_players()):
        _, best_response_utility = get_best_response(game, player, joint_policy)
        nash_conv += best_response_utility - expected_payoff[player]

    return nash_conv


def compute_nash_equilibrium(game: BaseNFGGame, method : str) -> JointPolicy:
    """Compute the Nash equilibrium of any 2-player 0-sum normal form game.

    Args:
        game (BaseNFGGame): the game to solve
        method (str): the method to use to compute the Nash equilibrium. It can be :
            - "LP" : formalize a 2-player 0-sum game as a linear program and solve it using scipy.optimize.linprog
            - "lagrangian" : for a 2-player game, if R0 and R1 are inversible, the Nash equilibrium can be computed using the Lagrangian method :
                pi_i_NE = normalization[ R_{-i}^{-1} @ ones((n_actions,)) ]
            
    Returns:
        JointPolicy: the Nash equilibrium of the game
    """
    
    if method == "LP":
        assert game.num_players() == 2, "This method only works for 2-player games"

        utility_matrix = game.get_utility_matrix()
        assert (
            utility_matrix[:, :, 0] == -utility_matrix[:, :, 1]
        ).all(), "This method only works for 0-sum games"

        n_actions = game.num_distinct_actions()

        # Define the objective function
        c_1 = [0] * n_actions[1] + [1]
        c_0 = [0] * n_actions[0] + [-1]

        # Define the constraints
        A_ub_0 = []
        b_0 = []
        A_eq_0 = []
        b_eq_0 = []
        
        A_ub_1 = []
        b_1 = []
        A_eq_1 = []
        b_eq_1 = []
        
        # Constraint for player 0
        for action in range(n_actions[0]):
            # R[a,:] @ p >= t
            A_ub_0.append(utility_matrix[:, action, 1].tolist() + [1])
            b_0.append(0)
            # p[a] >= 0
            constraint_vector = [0] * (n_actions[0] + 1)
            constraint_vector[action] = -1
            A_ub_0.append(constraint_vector)
            b_0.append(0)
        # The sum of the probabilities should be 1
        constraint_vector = [1] * n_actions[0] + [0]
        A_eq_0.append(constraint_vector)
        b_eq_0.append(1)
        # Solve the linear program
        pi_1_NE = linprog(c_0, A_ub=A_ub_0, b_ub=b_0, A_eq=A_eq_0, b_eq=b_eq_0).x[:-1]
        print(pi_1_NE)
        
        # Constraint for player 1
        for action in range(n_actions[1]):
            # R[:,a] @ p >= t
            A_ub_1.append(utility_matrix[action, :, 1].tolist() + [1])
            b_1.append(0)
            # p[a] >= 0
            constraint_vector = [0] * (n_actions[1] + 1)
            constraint_vector[action] = -1
            A_ub_1.append(constraint_vector)
            b_1.append(0)
        # The sum of the probabilities should be 1
        constraint_vector = [1] * n_actions[1] + [0]
        A_eq_1.append(constraint_vector)
        b_eq_1.append(1)
        # Solve the linear program
        pi_0_NE = linprog(c_1, A_ub=A_ub_1, b_ub=b_1, A_eq=A_eq_1, b_eq=b_eq_1).x[:-1]
        # Solve the linear program
        return np.array([pi_0_NE, pi_1_NE])

    elif method == "lagrangian":
        assert game.num_players() == 2, "This method only works for 2-player games"
        assert game.num_distinct_actions()[0] == game.num_distinct_actions()[1], "This method only works for games with the same number of actions for both players"
        
        R0 = game.get_utility_matrix()[:, :, 0]
        R1 = game.get_utility_matrix()[:, :, 1]

        if np.linalg.det(R0) == 0:
            R0 += np.eye(R0.shape[0]) * sys.float_info.epsilon
        if np.linalg.det(R1) == 0:
            R1 += np.eye(R1.shape[0]) * sys.float_info.epsilon
        
        ones = np.ones(shape=(game.num_distinct_actions()[0],))

        pi1 = np.linalg.inv(R0) @ ones
        pi1 /= np.sum(pi1)

        pi0 = np.linalg.inv(R1) @ ones
        pi0 /= np.sum(pi0)

        return np.array([pi0, pi1])
    
    else:
        raise ValueError(f"Unknown nash computation method {method}")