""" Stores the possible players

This file is Copyright (c) 2020 Mark Bedaywi
"""
from __future__ import annotations
import random
from typing import Optional
from Game import GameState, Player, GameTree, MoveNotLegalError


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
    value: Optional[float]
    children: list[MinimaxGameTree]

    def __init__(self, start_state: GameState, value: Optional[float] = None) -> None:
        super().__init__(start_state)
        self.value = value

    def find_values(self, depth: int = -1) -> None:
        """Runs the minimax algorithm to update the value of each node in the tree.

        If depth is not negative, then minimax is only run up to the specified depth"""
        if depth == 0 or self.root.winner() is not None:
            self.value = self.root.evaluate_position()
            return

        self.expand_root()
        assert self.children != []

        # Finds the value of each child
        for child in self.children:
            child.find_values(depth - 1)

        # Maximizes the value
        if self.root.turn:
            self.value = max([child.value for child in self.children])
        # Minimizes the value
        else:
            self.value = min([child.value for child in self.children])

    def expand_tree(self, state: GameState) -> None:
        """Add all children of state in self, if they are not already there.
        Adds a MinimaxGameTree instead of the generic GameTree

        Assumes that if some child is present, then all possible children are present.
        """
        if state == self.root:
            if self.children == []:
                self.children = [MinimaxGameTree(move) for move in self.root.legal_moves()]
        else:
            for child in self.children:
                child.expand_tree(state)

    def make_move(self, state: GameState) -> None:
        """Makes a move, updating root and children
        Updates the value of self.value

        Raises a MoveError if move not in children
        """
        for child in self.children:
            if child.root.previous_move == state.previous_move:
                self.children = child.children
                self.root = state
                self.value = child.value

                return

        raise MoveNotLegalError(str(state.previous_move))


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
        """Return the optimal move from the game state in self.game_tree.root

        Assumes the game is not over, that is, assumes there are possible
        legal moves from this position
        """
        turn = self.game_tree.root.turn
        self.game_tree.find_values(self.depth)

        possible_moves = self.game_tree.children
        best_move = possible_moves[0]
        for move in possible_moves:
            # If it is player 1's turn, maximise
            if move.value is None:
                breakpoint()
            if turn and move.value > best_move.value:
                best_move = move
            # If it is player 2's turn, minimise
            elif not turn and move.value < best_move.value:
                best_move = move

        return best_move.root
