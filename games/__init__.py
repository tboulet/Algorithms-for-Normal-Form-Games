from typing import Dict, Type

from games.base_nfg_game import BaseNFGGame
from games.kuhn_poker import KuhnPokerNFG
from games.matching_pennies import MatchingPennies
from games.rock_paper_scissor import RockPaperScissor

game_name_to_nfg_solver: Dict[str, Type[BaseNFGGame]] = {
    "matching_pennies": MatchingPennies,
    "kuhn_poker": KuhnPokerNFG,
    "rock_paper_scissor": RockPaperScissor,
}
