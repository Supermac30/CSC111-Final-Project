""" Stores the Abstract Class for Games and some simple games to test algorithms on.

This file is Copyright (c) 2020 Mark Bedaywi
"""
from __future__ import annotations
from typing import Any, Optional, Tuple

import pygame


class GameState:
    """An abstract class for representing a specific state of a game.

    Instance Attributes:
        - turn: Is True if it is player 1's turn and False otherwise
        - previous_move: Holds the previous move made. This is None if no move has been made yet.
    """
    turn: bool
    previous_move: Any

    def evaluate_position(self, hueristic_type: int = 0) -> float:
        """Return an evaluation of the current position. This is a float between -1 and 1
        where 0 is a tie, 1 is a win for

        huersitic_type is used when multiple different heuristics can be chosen from"""
        raise NotImplementedError

    def is_legal(self, move: Any) -> bool:
        """Return whether the next move is legal from the game state in self"""
        raise NotImplementedError

    def make_move(self, move: Any) -> bool:
        """Play the move move if it is legal

        Returns True if the move is legal and False otherwise.
        """
        raise NotImplementedError

    def change_state(self, new_state: GameState) -> bool:
        """Change the current state to new_state.

        Returns True if the state legally follows from self, and False otherwise.
        """
        raise NotImplementedError

    def winner(self) -> str:
        """Return a string detailing the winner, and None if the game is not over in the game_state in self."""
        raise NotImplementedError

    def legal_moves(self) -> list[GameState]:
        """Return a list of all possible legal moves from self"""
        raise NotImplementedError

    def equal(self, game_state: GameState):
        """Return whether self is equal to game_state"""
        raise NotImplementedError

    def display(self, screen: pygame.display) -> None:
        """An abstract method for displaying the current game state"""
        raise NotImplementedError


class Game:
    """ An abstract class for holding games.

    Instance Attributes:
        - history: Stores the moves performed by players
        - player1: Holds the first player
        - player2: Holds the second player
    """
    # Private Instance Attributes
    #   - game_state: Stores the current game state
    _game_state: GameState

    history: list[GameState]
    player1: Player
    player2: Player

    def __init__(self, player1: Player, player2: Player, game_state: GameState) -> None:
        self._game_state = game_state
        self.history = []
        self.player1 = player1
        self.player2 = player2

    def play_game(self) -> Tuple[Tuple[bool, int], list[GameState]]:
        """Plays a single game.

        This returns a tuple
        where the first element is if there is a tie or a winner,
        and the second is the id of the winner, and the history of moves
        """
        player = 0
        previous_state = None
        while self.winner() is None:
            if player == 0:
                new_state = self.player1.make_move(previous_state)
                player = 1
            else:
                new_state = self.player2.make_move(previous_state)
                player = 0
            self.change_state(new_state)

            previous_state = new_state
        return self.winner(), self.history

    def legal_moves(self) -> list[GameState]:
        """Return a list of legal moves from this position"""
        return self._game_state.legal_moves()

    def make_move(self, move: Any, check_legal: bool = True) -> None:
        """Change the current game state.

        If check_legal is true, an error is raised if the move is not legal.
        This can be made false to save time.
        """
        if not check_legal or self._game_state.is_legal(move):
            self.history.append(self._game_state)
            self._game_state.make_move(move)
        else:
            raise MoveNotLegalError(str(move))

    def change_state(self, new_state: GameState, check_legal: bool = True) -> None:
        """Change the current game state.

        If check_legal is true, an error is raised if the move is not legal.
        This can be made false to save time.
        """
        if check_legal:
            if all(not new_state.equal(legal_move) for legal_move in self.legal_moves()):
                raise MoveNotLegalError(str(new_state.previous_move))

        self.history.append(self._game_state)
        self._game_state = new_state

    def winner(self) -> Optional[Tuple[bool, int]]:
        """An abstract method returning a tuple where the first element is
        true if the game has a winner and false if it has a tie.
        The second element is the id of the player who won, or 0 if there is a tie.

        None is returned if the game is not yet over.
        """
        raise NotImplementedError


class Player:
    """An abstract class storing player methods

    Instance Attributes:
        - id: An integer storing the 'name' of the player for identification.
            Useful for when the number of players is greater than two.
        - game_tree: Holds the GameTree object the player uses to make decisions
    """
    id: int
    game_tree: GameTree

    def __init__(self, id_num: int, game_tree: GameTree) -> None:
        self.id = id_num
        self.game_tree = game_tree

    def make_move(self, opponent_move: Optional[GameState]) -> GameState:
        """Make a move from the current game state and previous move made by the
        opponent, and update the game tree.

        opponent_move is None if no move has been played yet.
        """
        # If there are no children of the root, add them
        self.game_tree.expand_root()

        if opponent_move is not None:
            self.game_tree.make_move(opponent_move)

        # If there are no children of the root, add them
        self.game_tree.expand_root()

        move_chosen = self.choose_move()

        self.game_tree.make_move(move_chosen)
        return move_chosen

    def choose_move(self) -> GameState:
        """An abstract method that chooses a move to play using the game tree in self
        """
        raise NotImplementedError


class GameTree:
    """Stores the game tree of an arbitrary game
    This is used by the various AIs to make decisions.

    Instance Attributes:
        - root: Holds the GameState in the root of self
        - children: Holds all subtrees of self connected to the root
    """
    root: GameState
    children: list[GameTree]

    def __init__(self, start_state: GameState) -> None:
        self.root = start_state
        self.children = []

    def find_children(self, state: GameState) -> list[GameTree]:
        """Return all children of state in self

        Returns an empty list if state is not in self
        """
        if state == self.root:
            return self.children

        for child in self.children:
            if state == child.root:
                return child.children

        for child in self.children:
            children = child.find_children(state)
            if children is not None:
                return children

        return []

    def expand_tree(self, state: GameState) -> None:
        """Add all children of state in self, if they are not already there.

        Assumes that if some child is present, then all possible children are present.
        """
        if state == self.root:
            if self.children == []:
                self.children = [GameTree(move) for move in self.root.legal_moves()]
        else:
            for child in self.children:
                child.expand_tree(state)

    def expand_root(self) -> None:
        """Creates children if there aren't any"""
        self.expand_tree(self.root)

    def make_move(self, state: GameState) -> None:
        """Makes a move, updating root and children

        Raises a MoveError if move not in children
        """
        children_states = [child.root.previous_move for child in self.children]

        if state.previous_move not in children_states:
            raise MoveNotLegalError(str(state.previous_move))

        self.children = self.find_children(state)
        self.root = state


class MoveNotLegalError(Exception):
    """The Error that is raised when a move that is
    attempted is not legal.

    Instance Attributes:
        - move: The move that caused the error
    """
    move: GameState

    def __init__(self, move: str) -> None:
        self.message = "The move " + move + " is not legal in this position."
        super().__init__(self.message)
