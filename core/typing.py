from typing import List

Policy = List[float]
JointPolicy = List[Policy]  # if p is of type Policy, then p[i][a] = p_i(a)
