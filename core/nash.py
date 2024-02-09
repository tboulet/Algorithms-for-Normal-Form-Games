from games.base_nfg_game import BaseNFGGame
from core.typing import JointPolicy, Policy
from typing import List

import numpy as np
from scipy.optimize import linprog
import copy


def get_expected_payoff(game: BaseNFGGame, joint_policy: JointPolicy) -> np.ndarray:
    utility_matrix = game.get_utility_matrix()

    expected_payoff = (
        utility_matrix * np.array(joint_policy[0])[:, np.newaxis, np.newaxis]
    )
    expected_payoff = (
        expected_payoff * np.array(joint_policy[1])[np.newaxis, :, np.newaxis]
    )

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
    utility_matrix = game.get_utility_matrix()

    if player == 0:
        weighted_utility = (
            utility_matrix[:, :, 0] * np.array(joint_policy[1])[:, np.newaxis]
        )
        weighted_utility = np.sum(weighted_utility, axis=0)
    else:
        weighted_utility = (
            utility_matrix[:, :, 1] * np.array(joint_policy[0])[np.newaxis, :]
        )
        weighted_utility = np.sum(weighted_utility, axis=1)

    max_utility = np.max(weighted_utility)

    return np.nonzero(weighted_utility == max_utility)[0], max_utility


def compute_nash_conv(game: BaseNFGGame, joint_policy: JointPolicy) -> float:
    nash_conv = 0.0
    expected_payoff = get_expected_payoff(game, joint_policy)

    for player in range(game.num_players()):
        _, best_response_utility = get_best_response(game, player, joint_policy)
        nash_conv += best_response_utility - expected_payoff[player]

    return nash_conv


def compute_nash_equilibrium(game: BaseNFGGame) -> JointPolicy:
    """Compute the Nash equilibrium of any 2-player 0-sum normal form game.

    Args:
        game (BaseNFGGame): the game to solve

    Returns:
        JointPolicy: the Nash equilibrium of the game
    """
    assert game.num_players() == 2, "This function only works for 2-player games"

    utility_matrix = game.get_utility_matrix()
    assert (
        utility_matrix[:, :, 0] == -utility_matrix[:, :, 1]
    ).all(), "This function only works for 0-sum games"

    n_actions = game.num_distinct_actions()

    # Define the objective function
    c_1 = [0] * n_actions + [1]
    c_0 = [0] * n_actions + [-1]

    # Define the constraints
    A_ub_0 = []
    A_ub_1 = []
    b = []
    A_eq = []
    b_eq = []
    # Each action should get a payoff inferior to w
    for action in range(n_actions):
        # R[a,:] @ p >= t
        A_ub_0.append(utility_matrix[:, action, 1].tolist() + [1])
        # R[:,a] @ p <= w
        A_ub_1.append(utility_matrix[action, :, 0].tolist() + [-1])
        b.append(0)
    # p[a] >= 0
    for action in range(n_actions):
        constraint_vector = [0] * (n_actions + 1)
        constraint_vector[action] = -1
        A_ub_0.append(constraint_vector)
        A_ub_1.append(constraint_vector)
        b.append(0)
    # The sum of the probabilities should be 1
    constraint_vector = [1] * n_actions + [0]
    A_eq.append(constraint_vector)
    b_eq.append(1)

    # Solve the linear program
    pi_1_NE = linprog(c_1, A_ub=A_ub_1, b_ub=b, A_eq=A_eq, b_eq=b_eq).x[:-1]
    pi_0_NE = linprog(c_0, A_ub=A_ub_0, b_ub=b, A_eq=A_eq, b_eq=b_eq).x[:-1]
    return np.array([pi_0_NE, pi_1_NE])
