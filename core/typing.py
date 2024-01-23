from typing import List, Tuple, Dict, Callable, Union, Optional, Any, Sequence, Iterable, TypeVar, Generic

Policy = List[float]
JointPolicy = List[Policy]   # if p is of type Policy, then p[i][a] = p_i(a)
