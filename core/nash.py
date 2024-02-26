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


def solve_nash_lp_formulation(
    reward_matrix: np.ndarray,
    ) -> Policy:
    """Solve the Nash Equilibrium LP formulation for a 2-player 0-sum game,
    for the first player.

    Args:
        reward_matrix (np.ndarray): the reward matrix

    Returns:
        Policy: the Nash equilibrium policy of the adversary
    """
    n0, n1 = reward_matrix.shape
    A = []
    b = []
    A_eq = []
    b_eq = []
    
    # Define the objective function : we try to minimize t (the lower bound on any expected result for player 0)
    c = [0] * n1 + [-1]
    
    # Constraint : lower bound
    for a0 in range(n0):
        constraint_vector = (-reward_matrix[a0, :]).tolist() + [1]
        A.append(constraint_vector)
        b.append(0)
        
    # Constraint : sum of the probabilities should be 1
    constraint_vector = [1] * n1 + [0]
    A_eq.append(constraint_vector)
    b_eq.append(1)
    
    # Constraint : (almost) positive probabilities
    for a1 in range(n1):
        constraint_vector = [0] * (n1 + 1)
        constraint_vector[a1] = -1
        A.append(constraint_vector)
        b.append(-sys.float_info.epsilon)
        
    # Solve the linear program
    pi_adv_NE = linprog(c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq).x[:-1]
    return pi_adv_NE

def compute_nash_equilibrium(game: BaseNFGGame, method : str) -> JointPolicy:
    """Compute the Nash equilibrium of any 2-player 0-sum normal form game.
    If error during computation, return None.
    
    Args:
        game (BaseNFGGame): the game to solve
        method (str): the method to use to compute the Nash equilibrium. It can be :
            - "LP" : formalize a 2-player 0-sum game as a linear program and solve it using scipy.optimize.linprog
            - "lagrangian" : for a 2-player game, if R0 and R1 are inversible, the Nash equilibrium can be computed using the Lagrangian method :
                pi_i_NE = normalization[ R_{-i}^{-1} @ ones((n_actions,)) ]
            
    Returns:
        JointPolicy: the Nash equilibrium of the game
    """
    try :
        if method == "LP":
            assert game.num_players() == 2, "This method only works for 2-player games"

            utility_matrix = game.get_utility_matrix()
            assert (
                utility_matrix[:, :, 0] == -utility_matrix[:, :, 1]
            ).all(), "This method only works for 0-sum games"

            pi_1_NE = solve_nash_lp_formulation(reward_matrix=utility_matrix[:, :, 0])
            pi_0_NE = solve_nash_lp_formulation(reward_matrix=-utility_matrix[:, :, 1].T)
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
        
    except Exception as e:
        print(f"Error while computing the Nash equilibrium using the method {method} : {e}")
        return