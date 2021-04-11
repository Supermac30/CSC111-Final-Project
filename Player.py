""" Stores the possible players

This file is Copyright (c) 2020 Mark Bedaywi
"""
from __future__ import annotations
import random

from Game import GameState, Game, Player, GameTree


class RandomPlayer(Player):
    """A player that makes random moves"""
    def choose_move(self) -> GameState:
        """Return a random move from the game state"""
        possible_moves = [child.root for child in self.game_tree.children]

        return random.choice(possible_moves)


class MinimaxGameTree(GameTree):
    """A GameTree that stores the value of the root for the purpose of minimax

     Instance Attributes:
        - root: Holds the GameState in the root of self.
        - value: Holds the value of the root state. This is None if the value has not been calculated yet.
        - children: Holds all subtrees of self connected to the root.
    """
    root: GameState
    value: float
    children: list[MinimaxGameTree]

    def __init__(self, start_state: GameState, value: float = 0) -> None:
        super().__init__(start_state)
        self.value = value

    def find_values(self, depth: int = -1) -> None:
        """Runs the minimax algorithm to update the value of each node in the tree.

        If depth is not negative, then minimax is only run up to the specified depth"""
        if depth == 0:
            self.value = self.root.evaluate_position()
            return

        self.expand_root()
        if self.children == []:  # No legal moves are available, and so the game is over
            self.value = self.root.evaluate_position()
            return

        # Finds the value of each child
        for child in self.children:
            child.find_values(depth - 1)

        # Maximizes the value
        if self.root.turn:
            self.value = max([child.value for child in self.children])
        # Minimizes the value
        else:
            self.value = min([child.value for child in self.children])


class MinimaxPlayer(Player):
    """A player that chooses the optimal move using the minimax algorithm

    Instance Attributes:
        - id: An integer storing the 'name' of the player for identification.
            Useful for when the number of players is greater than two.
        - game_tree: Holds the GameTree object the player uses to make decisions
        - depth: Holds the depth that the search will be made to
    """
    id: int
    game_tree: MinimaxGameTree
    depth: int

    def __init__(self, id_num: int, game_tree: MinimaxGameTree, depth: int = -1) -> None:
        super().__init__(id_num, game_tree)
        self.depth = depth

    def choose_move(self) -> GameState:
        """Return the optimal move from the game state in self.game_tree.root"""
        turn = self.game_tree.root.turn
        if self.game_tree.value is None:
            self.game_tree.find_values(self.depth)

        # If it is player 1's turn, maximise
        if turn:
            best_move = max((state for state in self.game_tree.children), key=lambda n: n.value)
        # Else, minimize
        else:
            best_move = min((state for state in self.game_tree.children), key=lambda n: n.value)

        return best_move.root
