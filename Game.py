""" Stores the Abstract Class for Games and some simple games to test algorithms on.

This file is Copyright (c) 2020 Mark Bedaywi
"""
from __future__ import annotations
from typing import Any, Optional, Tuple, Type, List

import pygame


class GameState:
    """An abstract class for representing a specific state of a game.

    Instance Attributes:
        - turn: Is True if it is player 1's turn and False otherwise
        - previous_move: Holds the previous move made. This is None if no move has been made yet.
        - game_type: Holds the type of game
        - board: Holds the board representing the game state
    """
    turn: bool
    previous_move: Any
    game_type: Type[Game]
    board: list

    def change_state(self, new_state: GameState) -> bool:
        """Change the current state to new_state.

        Returns True if the state legally follows from self, and False otherwise.
        """
        move = new_state.previous_move
        return self.make_move(move)

    def evaluate_position(self, heuristic_type: int = 0) -> float:
        """Return an evaluation of the current position. This is a float between -1 and 1
        where 0 is a tie, 1 is a win for player 1, and -1 is a win for player 2.

        heuristic_type is used when multiple different heuristics can be chosen from.

        When heuristic_type_type is set to 0, when the game is not over, a zero 0 is always returned."""
        raise NotImplementedError

    def vector_representation(self) -> List[float]:
        """Return a unique vector representation of the game state
        for the purpose of training Neural Networks"""
        raise NotImplementedError

    def is_legal(self, move: Any) -> bool:
        """Return whether the next move is legal from the game state in self"""
        raise NotImplementedError

    def make_move(self, move: Any, check_legal: bool = True) -> bool:
        """Play the move move if it is legal

        check_legal can be set to False to save time.

        Returns True if the move is legal and False otherwise.
        """
        raise NotImplementedError

    def winner(self) -> Optional[Tuple[bool, bool]]:
        """Return a tuple, where the first value is true if some player won,
        and the second value is true if player 1 won,

        Return None if the game is not over."""
        raise NotImplementedError

    def legal_moves(self) -> list[GameState]:
        """Return a list of all possible legal moves from self.

        Moves that are more likely to be better should be returned first,
        to help speed up the alpha-beta pruning.
        """
        raise NotImplementedError

    def equal(self, game_state: GameState):
        """Return whether self is equal to game_state"""
        raise NotImplementedError

    def display(self, screen: pygame.display) -> None:
        """An abstract method for displaying the current game state"""
        raise NotImplementedError

    def get_human_input(self, screen: pygame.display, click_loc: Optional[Tuple[int, int]]) -> Optional[GameState]:
        """Return a game state after a click at location click_loc. If no click has been performed yet,
        click_loc is set to None and None is returned.

        If the click is at an invalid input, None is returned."""
        raise NotImplementedError

    def __str__(self) -> str:
        """A unique string representation of the state for memoization and debugging"""
        raise NotImplementedError

    def copy(self) -> GameState:
        """Return a copy of self"""
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

    # TODO: Remove redundant _game_state attribute

    _game_state: GameState

    history: list[GameState]
    player1: Player
    player2: Player

    def __init__(self, player1: Player, player2: Player, game_state: GameState) -> None:
        self._game_state = game_state
        self.history = [game_state]
        self.player1 = player1
        self.player2 = player2

    def play_game(self, debug: bool = False) -> Tuple[Tuple[bool, bool], list[GameState]]:
        """Plays a single game.

        This returns a tuple
        where the first element is if there is a tie or a winner,
        and the second is True if player 1 won.

        If debug is true, print each game state

        Returns the history of moves as well.
        """
        if self._game_state.turn:
            player = 0
        else:
            player = 1

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

            if debug:
                print(previous_state)

        return self.winner(), self.history

    def play_games(self, n: int) -> Tuple[float, float]:
        """Play n games and return a Tuple where the first element is
        the proportion of times player 1 wins, and the second is the proportion of times player 2 wins"""
        player1_win = 0
        player2_win = 0
        for _ in range(n):
            winner = self.copy().play_game()[0]
            if winner[0]:
                if winner[1]:
                    player1_win += 1
                else:
                    player2_win += 1
        return (player1_win / n, player2_win / n)

    def play_with_human(self, is_player1: bool, screen: pygame.display) -> Tuple[Tuple[bool, bool], list[GameState]]:
        """Play a game with a human as player 1 if is_player1 is True and
        the human as player 2 otherwise.
        """
        if self._game_state.turn:
            player = 0
        else:
            player = 1
        self._game_state.display(screen)

        previous_state = None
        click_loc = None

        while self.winner() is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
                if event.type == pygame.MOUSEBUTTONUP:
                    click_loc = pygame.mouse.get_pos()

            if player == 0:
                if is_player1:
                    new_state = self._game_state.get_human_input(screen, click_loc)
                else:
                    new_state = self.player1.make_move(previous_state)

            else:
                if is_player1:
                    new_state = self.player2.make_move(previous_state)
                else:
                    new_state = self._game_state.get_human_input(screen, click_loc)

            # If a move has been made
            if new_state is not None:
                player = 1 - player  # change the player from 1 to 0 or vice versa
                self.change_state(new_state)

                new_state.display(screen)

                previous_state = new_state

        return self.winner(), self.history

    def legal_moves(self) -> list[GameState]:
        """Return a list of legal moves from this position"""
        return self._game_state.legal_moves()

    def make_move(self, move: Any, check_legal: bool = False) -> None:
        """Change the current game state.

        If check_legal is true, an error is raised if the move is not legal.
        This can be made false to save time.
        """
        if not check_legal or self._game_state.is_legal(move):
            self.history.append(self._game_state)
            self._game_state.make_move(move)
        else:
            raise MoveNotLegalError(str(move))

    def change_state(self, new_state: GameState, check_legal: bool = False) -> None:
        """Change the current game state.

        If check_legal is true, an error is raised if the move is not legal.
        This can be made false to save time.
        """
        if check_legal:
            if all(not new_state.equal(legal_move) for legal_move in self.legal_moves()):
                raise MoveNotLegalError(str(new_state.previous_move))

        self.history.append(new_state)
        self._game_state = new_state

    def winner(self) -> Optional[Tuple[bool, bool]]:
        """Return a tuple, where the first value is true if some player won,
        and the second value is true if player 1 won,

        Return None if the game is not over."""
        return self._game_state.winner()

    def copy(self) -> Game:
        """Return a copy of self"""
        raise NotImplementedError


class Player:
    """An abstract class storing player methods

    Instance Attributes:
        - game_tree: Holds the GameTree object the player uses to make decisions
    """
    game_tree: GameTree

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

    def copy(self) -> Player:
        """Return a copy of self"""
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
        for child in self.children:
            if child.root.previous_move == state.previous_move:
                self.children = child.children
                self.root = state
                return

        raise MoveNotLegalError(str(state.previous_move))

    def copy(self) -> GameTree:
        """Return a copy of self"""
        raise NotImplementedError


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
