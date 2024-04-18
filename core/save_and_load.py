

import os
from typing import List
import numpy as np
from core.typing import JointPolicy


def save_joint_policy(
            joint_policy : JointPolicy,
            paths : List[str],
            verbose : bool = False,
        ) -> None:
    """Save a joint policy to a file.

    Args:
        joint_policy (JointPolicy): the joint policy to save, i.e. a list of policies where each policy is a numpy array (one per player)
        paths (List[str]): the paths where to save the joint policy
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
        for i, policy in enumerate(joint_policy):
            np.save(f"{path}/policy_{i}.npy", policy)
        if verbose:
            print(f"Joint policy saved at {path}")
        

def load_joint_policy(
            path : str,
            verbose : bool = False,
        ) -> JointPolicy:
    """Load a joint policy from a file.

    Args:
        path (str): the path to the file where the joint policy is saved
    """
    joint_policy = []
    shapes = []
    i = 0
    while os.path.exists(f"{path}/policy_{i}.npy"):
        policy = np.load(f"{path}/policy_{i}.npy")
        joint_policy.append(policy)
        shapes.append(policy.shape)
        i += 1
    if verbose:
        print(f"Joint policy loaded from {path}. Shapes: {shapes}")
    return joint_policy