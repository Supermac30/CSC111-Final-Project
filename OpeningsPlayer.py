"""Holds the player that uses openings for various games

This file is Copyright (c) 2020 Mark Bedaywi
"""
from __future__ import annotations

import random
from typing import Tuple
import Reversi

from Game import Player, GameTree, GameState


class ReversiOpeningsGameTree(GameTree):
    """The game tree that uses game data to make moves by memorizing good moves"""
    children: list[ReversiOpeningsGameTree]
    root: Reversi.ReversiGameState

    def __init__(self, start_state: Reversi.ReversiGameState) -> None:
        super().__init__(start_state)
        self.build_tree()

    def build_tree(self) -> None:
        """Read the relevant data set and build the game tree in self"""
        for i in range(1, 51):
            filename = 'data/reversi_games/' + str(i) + '_w.txt'
            file = open(filename)
            lines = file.readlines()
            moves = []
            previous_player = 0
            for line in lines:
                values = [int(value) for value in line.split()]
                # If a pass has been played
                if values[0] == previous_player:
                    moves.append(None)
                else:
                    previous_player = 1 - previous_player

                moves.append((values[1], values[2]))

            self.add_move_sequence(moves)

    def add_move_sequence(self, moves: list[Tuple[int, int]]) -> None:
        """Add a sequence of moves to self.

        Preconditions:
            - self.root.make_move(moves[0], True) == True
        """
        possible_moves = [child.root.previous_move for child in self.children]
        if moves[0] in possible_moves:
            position = possible_moves.index(moves[0])
            child = self.children[position]
        else:
            new_move = self.root.copy()
            new_move.make_move(moves[0], False)
            child = ReversiOpeningsGameTree(new_move)
            self.children.append(child)

        child.add_move_sequence(moves[1:])

    def expand_tree(self, state: GameState) -> None:
        """Doesn't expand the tree with legal moves.
        """
        return None

    def copy(self) -> GameTree:
        """Return a copy of self"""
        return ReversiOpeningsGameTree(self.root.copy())


class ReversiOpeningsPlayer(Player):
    """The player that plays the opening in Reversi as long as possible,
    before reverting to some other player.

    Instance Attributes:
        - start_state: Holds the state the game starts with
        - game_tree: The game tree with winning games
        - default_player: stores the player that will play
            once the opening is exhausted.
    """
    start_state: Reversi.ReversiGameState
    game_tree: ReversiOpeningsGameTree
    default_player: Player

    def __init__(self, start_state: Reversi.ReversiGameState, default_player: Player,
                 game_tree: ReversiOpeningsGameTree = None):
        self.start_state = start_state
        self.default_player = default_player
        if game_tree is None:
            self.game_tree = ReversiOpeningsGameTree(self.start_state)
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

    def copy(self) -> ReversiOpeningsPlayer:
        """Return a copy of self"""
        return ReversiOpeningsPlayer(
            self.start_state,
            self.default_player,
            self.game_tree
        )
