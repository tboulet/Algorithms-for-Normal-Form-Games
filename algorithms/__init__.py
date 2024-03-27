from typing import Dict, Type

from algorithms.base_nfg_algorithm import BaseNFGAlgorithm
from algorithms.forel import Forel
from algorithms.iterated_forel_lyapunov import IteratedForel
from algorithms.policy_gradient import PolicyGradient
from algorithms.population_alternating_lyapunov_forel import PopulationAlternatingLyapunovForel
from algorithms.population_forel import PopulationForel
from algorithms.population_iterated_lyapunov_forel import PopulationIteratedLyapunovForel
from algorithms.pdl_forel import PDLForel

algo_name_to_nfg_solver: Dict[str, Type[BaseNFGAlgorithm]] = {
    "Policy Gradients": PolicyGradient,
    "IL-Forel": IteratedForel,
    "FoReL": Forel,
    "Population-FoReL": PopulationForel,
    "PIL-FoReL" : PopulationIteratedLyapunovForel,
    "PAL-FoReL" : PopulationAlternatingLyapunovForel,
    "pdl_forel" : PDLForel
}
