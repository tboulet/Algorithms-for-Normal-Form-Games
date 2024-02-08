from games.base_nfg_game import BaseNFGGame
from core.typing import JointPolicy, Policy
from typing import List

import numpy as np
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
            utility_matrix * np.array(joint_policy[1])[:, np.newaxis, np.newaxis]
        )
        weighted_utility = np.sum(weighted_utility, axis=0)
    else:
        weighted_utility = (
            utility_matrix * np.array(joint_policy[0])[np.newaxis, :, np.newaxis]
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
