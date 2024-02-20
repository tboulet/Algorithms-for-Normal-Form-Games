import itertools
import random
from typing import Dict, List, Tuple
from enum import Enum
from core.typing import JointPolicy
import numpy as np
from games.base_nfg_game import BaseNFGGame


class KuhnPokerAction(str, Enum):
    CHECK = "CHECK"
    BET = "BET"
    FOLD = "FOLD"
    CALL = "CALL"


class KuhnPokerCards(int, Enum):
    KING = 2
    QUEEN = 1
    JACK = 0


# Types
Plan = Tuple[
    KuhnPokerAction
]  # a plan is a n-uple of operations that will later be used in the game tree
Strategy = Dict[KuhnPokerCards, Plan]  # a strategy maps a card to a plan

cards = list(KuhnPokerCards)


def possible_plans_to_action_index_to_strategy(
    possible_plans: List[Plan],
) -> Dict[int, Strategy]:
    """Returns a dictionary that maps an action index to a strategy conditioned on the card of the player"""
    action_index_to_strategy: Dict[int, Strategy] = {}
    for action_index, (plan_1, plan_2, plan_3) in enumerate(
        itertools.product(possible_plans, possible_plans, possible_plans)
    ):
        strategy: Strategy = {
            KuhnPokerCards.JACK: plan_1,
            KuhnPokerCards.QUEEN: plan_2,
            KuhnPokerCards.KING: plan_3,
        }
        action_index_to_strategy[action_index] = strategy

    return action_index_to_strategy


def sign(x: float) -> float:
    return 0.0 if abs(x) == 0 else x / abs(x)


class KuhnPokerNFG(BaseNFGGame):
    """A Normal Form Game version of the Kuhn Poker game, where the actions consist of establishing in advance the strategy, and the reward is stochastically drawn from the game tree."""

    # Possible plans for each player
    Player1Plans = [
        # check first then fold if the other player bets
        (KuhnPokerAction.CHECK, KuhnPokerAction.FOLD),
        # check first then call if the other player bets
        (KuhnPokerAction.CHECK, KuhnPokerAction.CALL),
        # bet first
        (KuhnPokerAction.BET,),
    ]

    Player2Plans = [
        # check if the other player checks and fold if the other player bets
        (KuhnPokerAction.CHECK, KuhnPokerAction.FOLD),
        # check if the other player checks and call if the other player bets
        (KuhnPokerAction.CHECK, KuhnPokerAction.CALL),
        # bet if the other player checks and fold if the other player bets
        (KuhnPokerAction.BET, KuhnPokerAction.FOLD),
        # bet if the other player checks and call if the other player bets
        (KuhnPokerAction.BET, KuhnPokerAction.CALL),
    ]

    actions_index_to_action_1 = possible_plans_to_action_index_to_strategy(Player1Plans)
    actions_index_to_action_2 = possible_plans_to_action_index_to_strategy(Player2Plans)

    utility_matrix = None

    def get_game_outcome(
        self, plan1: Plan, plan2: Plan, card1: KuhnPokerCards, card2: KuhnPokerCards
    ) -> List[float]:
        if plan1[0] == KuhnPokerAction.CHECK:
            if plan2[0] == KuhnPokerAction.CHECK:
                reward_for_player1 = sign(card1 - card2)
            elif plan2[0] == KuhnPokerAction.BET:
                if plan1[1] == KuhnPokerAction.CALL:
                    reward_for_player1 = 2 * (sign(card1 - card2))
                elif plan1[1] == KuhnPokerAction.FOLD:
                    reward_for_player1 = -1
                else:
                    raise ValueError(f"Unknown action {plan1[1]}")
            else:
                raise ValueError(f"Unknown action {plan2[0]}")
        elif plan1[0] == KuhnPokerAction.BET:
            if plan2[1] == KuhnPokerAction.FOLD:
                reward_for_player1 = 1
            elif plan2[1] == KuhnPokerAction.CALL:
                reward_for_player1 = 2 * sign(card1 - card2)
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
        state_player1 = random.choice(cards)
        state_player2 = random.choice([c for c in cards if c != state_player1])

        # Get the plan of both players
        plan_player1 = action_player1[state_player1]
        plan_player2 = action_player2[state_player2]

        # Get the reward
        return self.get_game_outcome(
            plan_player1, plan_player2, state_player1, state_player2
        )

    def num_distinct_actions(self) -> List[int]:
        return [
            len(self.actions_index_to_action_1),
            len(self.actions_index_to_action_2),
        ]

    def num_players(self) -> int:
        return 2

    def get_utility(
        self, action1: int, action2: int, memo_strat_card_rewards: Dict
    ) -> tuple[np.ndarray, Dict]:
        sum_reward = np.zeros(2)

        for card1 in cards:
            for card2 in cards:
                if card1 == card2:
                    continue

                plan1 = self.actions_index_to_action_1[action1][card1]
                plan2 = self.actions_index_to_action_2[action2][card2]
                game_instance = (plan1, plan2, card1, card2)

                if game_instance in memo_strat_card_rewards:
                    sum_reward += memo_strat_card_rewards[game_instance]
                else:
                    reward = np.array(self.get_game_outcome(plan1, plan2, card1, card2))
                    memo_strat_card_rewards[game_instance] = reward
                    sum_reward += reward

        return sum_reward / 6, memo_strat_card_rewards

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

        memo_strat_card_rewards = {}  # dict [(strat1, strat2, card1, card2)] -> rewards

        for action_1 in range(len(self.actions_index_to_action_1)):
            for action_2 in range(len(self.actions_index_to_action_2)):
                utility_matrix[action_1, action_2], memo_strat_card_rewards = (
                    self.get_utility(action_1, action_2, memo_strat_card_rewards)
                )

        self.utility_matrix = utility_matrix

        return utility_matrix
