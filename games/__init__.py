from typing import Dict, Type

from games.base_nfg_game import BaseNFGGame
from games.kuhn_poker import KuhnPokerNFG
from games.matching_pennies import MatchingPennies

game_name_to_nfg_solver: Dict[str, Type[BaseNFGGame]] = {
    "matching_pennies": MatchingPennies,
    "kuhn_poker": KuhnPokerNFG,
}
