from enum import Enum
from typing import List
import numpy as np

Policy = np.ndarray
JointPolicy = List[Policy]  # if p is of type Policy, then p[i][a] = p_i(a)


class DynamicMethod(str, Enum):
    RD = "rd"
    SOFTMAX = "softmax"


class QValueEstimationMethod(str, Enum):
    MC = "mc"
    MODEL_BASED = "model-based"


class Regularizer(str, Enum):
    ENTROPY = "entropy"
    L2 = "l2"
