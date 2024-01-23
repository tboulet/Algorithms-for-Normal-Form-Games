


from typing import List
from games.base_nfg_game import BaseNFGGame


class MatchingPennies(BaseNFGGame):
    
    def get_rewards(self, joint_action: List[int]) -> List[float]:
        if joint_action[0] == joint_action[1]:
            return [1, -1]
        else:
            return [-1, 1]
        
    def num_distinct_actions(self) -> int:
        return 2
    
    def num_players(self) -> int:
        return 2