from typing import Dict, Type

from algorithms.base_nfg_algorithm import BaseNFGAlgorithm
from algorithms.forel import Forel
from algorithms.iterated_forel import IteratedForel

algo_name_to_nfg_solver : Dict[str, Type[BaseNFGAlgorithm]] = {
    "iterated_forel" : IteratedForel,
    "forel" : Forel,
}