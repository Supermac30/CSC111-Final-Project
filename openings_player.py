"""Holds the player that uses openings for various games

This file is Copyright (c) 2020 Mark Bedaywi
"""
from __future__ import annotations

import random
from typing import Tuple, Optional
import reversi

from game import Player, GameTree, GameState


class ReversiOpeningsGameTree(GameTree):
    """The game tree that uses game data to make moves by memorizing good moves"""
    children: list[ReversiOpeningsGameTree]
    root: reversi.ReversiGameState

    def __init__(self, start_state: reversi.ReversiGameState,
                 initialise_tree: bool = True) -> None:
        super().__init__(start_state)
        if initialise_tree:
            self.build_tree()

    def build_tree(self) -> None:
        """Read the relevant data set and build the game tree in self"""
        for i in range(1, 51):
            filename = 'data/reversi_games/' + str(i) + '_w.txt'
            file = open(filename)
            lines = file.readlines()
            moves = []
            previous_player = 1
            for line in lines:
                # The data is of the form [player_id, x_coordinate, y_coordinate]
                values = [int(value) for value in line.split()]

                # If a pass has been played
                if values[0] == previous_player:
                    moves.append(None)
                else:
                    previous_player = 1 - previous_player

                # The order is flipped in the data set
                moves.append((values[2], values[1]))
            self.add_move_sequence(moves)

    def add_move_sequence(self, moves: list[Tuple[int, int]]) -> None:
        """Add a sequence of moves to self.

        Preconditions:
            - self.root.is_legal(moves[0])
        """
        # Base case
        if moves == []:
            return

        possible_moves = [child.root.previous_move for child in self.children]
        if moves[0] in possible_moves:
            position = possible_moves.index(moves[0])
            chosen_child = self.children[position]
        else:
            new_move = self.root.copy()
            if not new_move.make_move(moves[0], True):
                breakpoint()
            chosen_child = ReversiOpeningsGameTree(new_move, initialise_tree=False)
            self.children.append(chosen_child)

        chosen_child.add_move_sequence(moves[1:])

    def expand_tree(self, state: GameState) -> None:
        """Doesn't expand the tree with legal moves,
        sticking with the moves in the opening.
        """
        return None

    def make_move(self, state: reversi.ReversiGameState) -> None:
        """Makes a move, updating root and children.

        Raises a MoveError if move not in children.
        Preconditions:
            self.root.is_legal(state.previous_move)
        """
        for child in self.children:
            if child.root.previous_move == state.previous_move:
                self.children = child.children
                self.root = state
                return

        # If we get here, then the opponent has made a move not in our openings data base.
        # We then delete all children, forcing the ReversiOpeningsPlayer to use
        # the default_player
        self.children = []
        return

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
    start_state: reversi.ReversiGameState
    game_tree: GameTree
    default_player: Player

    def __init__(self, start_state: reversi.ReversiGameState, default_player: Player,
                 game_tree: GameTree = None) -> None:
        self.start_state = start_state
        self.default_player = default_player
        if game_tree is None:
            self.game_tree = ReversiOpeningsGameTree(self.start_state)
        else:
            self.game_tree = game_tree

    def make_move(self, opponent_move: Optional[reversi.ReversiGameState]) -> GameState:
        """Make a move from the current game state and previous move made by the
        opponent, and update the game tree.

        Here, default_player's game_tree is also updated.

        opponent_move is None if no move has been played yet.
        """
        self.default_player.game_tree.expand_root()

        if opponent_move is not None:
            self.game_tree.make_move(opponent_move)
            self.default_player.game_tree.make_move(opponent_move)

            self.default_player.game_tree.expand_root()

        move_chosen = self.choose_move()

        self.game_tree.make_move(move_chosen)
        self.default_player.game_tree.make_move(move_chosen)
        return move_chosen

    def choose_move(self) -> GameState:
        """Choose the move.

        Returns a random child of game_tree

        If game_tree reaches a leaf state that is not terminal,
        we have exhausted the opening. Then, self is changed into the new player
        """
        if self.game_tree.children == []:
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


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'max-line-length': 100,
        'disable': ['E1136']
    })
