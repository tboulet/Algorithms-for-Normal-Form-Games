from typing import Dict, List, Tuple
from core.typing import JointPolicy
import numpy as np
from games.base_nfg_game import BaseNFGGame


# Types
Plan = List[
    str
]  # a plan is a n-uple of operations that will later be used in the game tree
Strategy = Dict[int, Plan]  # a strategy map a

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


def possible_plans_to_action_index_to_strategy(
    possible_plans: List[Plan],
) -> Dict[int, Strategy]:
    n_plans = len(possible_plans)
    action_index_to_strategy: Dict[int, Strategy] = {}
    for idx_action_in_state_1 in range(n_plans):
        for idx_action_in_state_2 in range(n_plans):
            for idx_action_in_state_3 in range(n_plans):
                action_index = (
                    16 * idx_action_in_state_1
                    + 4 * idx_action_in_state_2
                    + idx_action_in_state_3
                )  # action_index is in [0, 63] and is the base 4 representation of the tuple (idx_action_in_state_1, idx_action_in_state_2, idx_action_in_state_3)
                strategy: Strategy = {
                    0: possible_plans[idx_action_in_state_1],
                    1: possible_plans[idx_action_in_state_2],
                    2: possible_plans[idx_action_in_state_3],
                }
                action_index_to_strategy[action_index] = strategy
    return action_index_to_strategy


def int_to_reward(n: int) -> int:
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

    assert len(Player1Plans) == len(
        Player2Plans
    ), "The number of plans for both players must be the same in this implementation"
    actions_index_to_action_1 = possible_plans_to_action_index_to_strategy(Player1Plans)
    actions_index_to_action_2 = possible_plans_to_action_index_to_strategy(Player2Plans)

    utility_matrix = None

    def get_game_outcome(
        self, plan1: Plan, plan2: Plan, card1: str, card2: str
    ) -> List[float]:
        if plan1[0] == CHECK:
            if plan2[0] == CHECK:
                reward_for_player1 = int_to_reward(
                    cards_to_values[card1] - cards_to_values[card2]
                )
            elif plan2[0] == BET:
                if plan1[1] == CALL:
                    reward_for_player1 = 2 * int_to_reward(
                        cards_to_values[card1] - cards_to_values[card2]
                    )
                elif plan1[1] == FOLD:
                    reward_for_player1 = -1
                else:
                    raise ValueError(f"Unknown action {plan1[1]}")
            else:
                raise ValueError(f"Unknown action {plan2[0]}")
        elif plan1[0] == BET:
            if plan2[1] == FOLD:
                reward_for_player1 = 1
            elif plan2[1] == CALL:
                reward_for_player1 = 2 * int_to_reward(
                    cards_to_values[card1] - cards_to_values[card2]
                )
            else:
                raise ValueError(f"Unknown action {plan2[0]}")
        else:
            raise ValueError(f"Unknown action {plan1[0]}")

        return [reward_for_player1, -reward_for_player1]

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
        return self.get_game_outcome(
            plan_player1, plan_player2, state_player1, state_player2
        )

    def num_distinct_actions(self) -> int:
        return len(self.actions_index_to_action_1)

    def num_players(self) -> int:
        return 2

    def get_utility(
        self, action1: int, action2: int, memo_strat_card_rewards: Dict
    ) -> tuple[np.ndarray, Dict]:
        sum_reward = np.zeros(2)

        for card1 in cards:
            for card2 in cards:
                plan1 = self.actions_index_to_action_1[action1][cards.index(card1)]
                plan2 = self.actions_index_to_action_2[action2][cards.index(card2)]
                game_instance = (tuple(plan1), tuple(plan2), card1, card2)

                if game_instance in memo_strat_card_rewards:
                    sum_reward += memo_strat_card_rewards[game_instance]
                else:
                    reward = np.array(self.get_game_outcome(plan1, plan2, card1, card2))
                    memo_strat_card_rewards[game_instance] = reward
                    sum_reward += reward

        return sum_reward / 9, memo_strat_card_rewards

    def get_utility_matrix(self) -> np.ndarray:
        if self.utility_matrix is not None:
            return self.utility_matrix

        utility_matrix = np.zeros(
            (
                len(self.actions_index_to_action_1),
                len(self.actions_index_to_action_2),
                2,
            )
        )

        memo_strat_card_rewards = (
            {}
        )  # dict [(tuple(strat1), tuple(strat2), card1, card2)] -> rewards

        for action_1 in range(len(self.actions_index_to_action_1)):
            for action_2 in range(len(self.actions_index_to_action_2)):
                utility_matrix[action_1, action_2], memo_strat_card_rewards = (
                    self.get_utility(action_1, action_2, memo_strat_card_rewards)
                )

        self.utility_matrix = utility_matrix

        return utility_matrix
