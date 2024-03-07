from typing import Dict, Type

from algorithms.base_nfg_algorithm import BaseNFGAlgorithm
from algorithms.forel import Forel
from algorithms.iterated_forel_lyapunov import IteratedForel
from algorithms.policy_gradient import PolicyGradient
from algorithms.population_forel import PopulationForel
from algorithms.population_iterated_lyapunov_forel import PopulationIteratedLyapunovForel

algo_name_to_nfg_solver: Dict[str, Type[BaseNFGAlgorithm]] = {
    "policy_gradient": PolicyGradient,
    "iterated_forel_lyapunov": IteratedForel,
    "forel": Forel,
    "population_forel": PopulationForel,
    "PIL_FoReL" : PopulationIteratedLyapunovForel,
}
