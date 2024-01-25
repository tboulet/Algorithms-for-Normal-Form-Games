from typing import Dict, Type

from algorithms.base_nfg_algorithm import BaseNFGAlgorithm
from algorithms.forel import Forel
from algorithms.iterated_forel_lyapunov import IteratedForel
from algorithms.softmax_policy_gradient import SoftmaxPolicyGradient

algo_name_to_nfg_solver : Dict[str, Type[BaseNFGAlgorithm]] = {
    "softmax_policy_gradient" : SoftmaxPolicyGradient,
    "iterated_forel_lyapunov" : IteratedForel,
    "forel" : Forel,
}