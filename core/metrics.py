import numpy as np
from typing import List
from enum import Enum


class DistributionDistance(str, Enum):
    KL = "kl"
    L1 = "l1"
    L2 = "l2"


def kl_divergence(joint_pi: List[np.ndarray], nash_pi: List[np.ndarray]) -> float:
    return sum(
        np.sum(joint_pi[player] * np.log(joint_pi[player] / nash_pi[player]))
        for player in range(len(joint_pi))
    )


def l2_distance(joint_pi: List[np.ndarray], nash_pi: List[np.ndarray]) -> float:
    return sum(
        np.sum((joint_pi[player] - nash_pi[player]) ** 2)
        for player in range(len(joint_pi))
    )


def l1_distance(joint_pi: List[np.ndarray], nash_pi: List[np.ndarray]) -> float:
    return sum(
        np.sum(np.abs(joint_pi[player] - nash_pi[player]))
        for player in range(len(joint_pi))
    )


def get_distance_function(distance_name: str | DistributionDistance = "kl"):
    if not isinstance(distance_name, DistributionDistance):
        distance_name = DistributionDistance(distance_name)

    if distance_name == DistributionDistance.KL:
        return kl_divergence
    elif distance_name == DistributionDistance.L1:
        return l1_distance
    elif distance_name == DistributionDistance.L2:
        return l2_distance
    else:
        raise ValueError(f"Unknown distance function {distance_name}")
