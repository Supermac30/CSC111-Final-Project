"""Holds the player that uses openings for various games

This file is Copyright (c) 2020 Mark Bedaywi
"""
from __future__ import annotations

import random
from typing import Type, Optional

from Game import Player, GameTree, GameState


class OpeningsGameTree(GameTree):
    """The game tree that uses game data to make moves by memorizing good moves"""
    children: list[OpeningsGameTree]

    def __init__(self, start_state: GameState) -> None:
        super().__init__(start_state)
        self.build_tree()

    def build_tree(self) -> None:
        """Read the relevant data set and build the game tree in self"""

    def add_move_sequence(self, moves: list[list]) -> None:
        """Add a sequence of moves to self.

        Preconditions:
            - moves[0] is not the move in self
        """
        children_boards = [child.root.board for child in self.children]
        if moves[0] in children_boards:
            position = children_boards.index(moves[0])
            child = self.children[position]
        else:
            # TODO
            child = GameState
            pass

        child.add_move_sequence(moves[1:])

    def expand_tree(self, state: GameState) -> None:
        """Doesn't expand the tree with legal moves.
        """
        return None

    def copy(self) -> GameTree:
        """Return a copy of self"""


class OpeningsPlayer(Player):
    """The player that plays the opening as long as possible,
    before reverting to some other player.

    Instance Attributes:
        - start_state: Holds the state the game starts with
        - game_tree: The game tree with winning games
        - default_player: stores the player that will play
            once the opening is exhausted.
    """
    start_state: GameState
    game_tree: OpeningsGameTree
    default_player: Player

    def __init__(self, start_state: GameState, default_player: Player,
                 game_tree: OpeningsGameTree = None):
        self.start_state = start_state
        self.default_player = default_player
        if game_tree is None:
            self.game_tree = OpeningsGameTree(self.start_state)
        else:
            self.game_tree = game_tree

    def choose_move(self) -> GameState:
        """Choose the move.

        Returns a random child of game_tree

        If game_tree reaches a leaf state that is not terminal, we have exhausted the opening.
        Then, self is changed into the new player
        """
        if self.game_tree.children == []:
            self.default_player.game_tree.root = self.game_tree.root
            return self.default_player.choose_move()
        else:
            return random.choice(self.game_tree.children).root

    def copy(self) -> OpeningsPlayer:
        """Return a copy of self"""
        return OpeningsPlayer(
            self.start_state,
            self.default_player,
            self.game_tree
        )
