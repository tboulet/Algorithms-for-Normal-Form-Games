from typing import Dict, List, Tuple
import numpy as np
from games.base_nfg_game import BaseNFGGame



# Types
Plan = List[str]                # a plan is a n-uple of operations that will later be used in the game tree
Strategy = Dict[int, Plan]      # a strategy map a

# Operations
CHECK = "CHECK"
BET = "BET"
FOLD = "FOLD"
CALL = "CALL"


# Cards
KING = "KING"
QUEEN = "QUEEN"
JACK = "JACK"

cards = [KING, QUEEN, JACK]

cards_to_values = {
    KING: 2,
    QUEEN: 1,
    JACK: 0,
}




def possible_plans_to_action_index_to_strategy(possible_plans : List[Plan]) -> Dict[int, Strategy]:
    n_plans = len(possible_plans)
    action_index_to_strategy : Dict[int, Strategy] = {}
    for idx_action_in_state_1 in range(n_plans):
        for idx_action_in_state_2 in range(n_plans):
            for idx_action_in_state_3 in range(n_plans):
                action_index = 16 * idx_action_in_state_1 + 4 * idx_action_in_state_2 + idx_action_in_state_3  # action_index is in [0, 63] and is the base 4 representation of the tuple (idx_action_in_state_1, idx_action_in_state_2, idx_action_in_state_3)
                strategy : Strategy = {
                    0: possible_plans[idx_action_in_state_1],
                    1: possible_plans[idx_action_in_state_2],
                    2: possible_plans[idx_action_in_state_3],
                }
                action_index_to_strategy[action_index] = strategy
    return action_index_to_strategy


def int_to_reward(n : int) -> int:
    if n > 0:
        return 1
    elif n < 0:
        return -1
    else:
        return 0
    



class KuhnPokerNFG(BaseNFGGame):
    """A Normal Form Game version of the Kuhn Poker game, where the actions consist of establishing in advance the strategy, and the reward is stochastically drawn from the game tree."""
    
    # Possible plans for each player
    Player1Plans = [
        [CHECK, FOLD],
        [CHECK, CALL],
        [BET],
        [BET],  # 2nd BET plan for having the same number of plans for both players
    ]

    Player2Plans = [
        [CHECK, FOLD],
        [CHECK, CALL],
        [BET, FOLD],
        [BET, CALL],
    ]

    assert len(Player1Plans) == len(Player2Plans), "The number of plans for both players must be the same in this implementation"
    actions_index_to_action_1 = possible_plans_to_action_index_to_strategy(Player1Plans)
    actions_index_to_action_2 = possible_plans_to_action_index_to_strategy(Player2Plans)


    def get_rewards(self, joint_action: List[int]) -> List[float]:
        # Player 1 pick an action (i.e.) a strategy pi : s -> plan
        idx_action_player1 = joint_action[0]
        action_player1 = self.actions_index_to_action_1[idx_action_player1]
        
        # Player 2 pick an action (i.e.) a strategy pi : s -> plan
        idx_action_player2 = joint_action[1]
        action_player2 = self.actions_index_to_action_2[idx_action_player2]
        
        # State (i.e. card) is given by Dealer's card
        state_player1 = np.random.choice(cards)
        state_player2 = np.random.choice([c for c in cards if c != state_player1])
        
        # Get the plan of both players
        plan_player1 = action_player1[cards.index(state_player1)]
        plan_player2 = action_player2[cards.index(state_player2)]
        
        # Get the reward
        if plan_player1[0] == CHECK:
            if plan_player2[0] == CHECK:
                reward_for_player1 = int_to_reward(cards_to_values[state_player1] - cards_to_values[state_player2])
            elif plan_player2[0] == BET:
                if plan_player1[1] == CALL:
                    reward_for_player1 = 2 * int_to_reward(cards_to_values[state_player1] - cards_to_values[state_player2])
                elif plan_player1[1] == FOLD:
                    reward_for_player1 = -1
                else:
                    raise ValueError(f"Unknown action {plan_player1[1]}")
            else:
                raise ValueError(f"Unknown action {plan_player2[0]}")
        elif plan_player1[0] == BET:
            if plan_player2[1] == FOLD:
                reward_for_player1 = 1
            elif plan_player2[1] == CALL:
                reward_for_player1 = 2 * int_to_reward(cards_to_values[state_player1] - cards_to_values[state_player2])
            else:
                raise ValueError(f"Unknown action {plan_player2[0]}")
        else:
            raise ValueError(f"Unknown action {plan_player1[0]}")
            
        return [reward_for_player1, -reward_for_player1]
        
    def num_distinct_actions(self) -> int:
        return 2 ** len(self.Player1Plans)
    
    def num_players(self) -> int:
        return 2