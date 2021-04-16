"""Holds the Reversi Game

This file is Copyright (c) 2020 Mark Bedaywi
"""
from __future__ import annotations
from typing import Optional, Tuple, Type, List

from Game import Game, GameState
import pygame
import copy


class ReversiGameState(GameState):
    """Stores the game state of a TicTacToe game

    Instance Attributes:
        - n: The dimension of the game state. n must be even.
        - game_state: A 2D nxn list storing the object in each position in the game.
            A 1 is placed if a black piece is in the location, 0 if it is a white piece and -1 if it is empty.
        - turn: Stores the turn of the player. This is true if it is X's turn and False otherwise.
        - previous_move: Stores the previous move made. This is None if no move has been made yet.
        - has_passed: Stores whether the previous player has passed. If both players pass, the game is over.
    """
    n: int
    board: list[list[int]]
    turn: bool
    previous_move: Optional[Tuple[int, int]]
    has_passed: bool

    def __init__(self, n: int = 8, game_state: Optional[ReversiGameState] = None, has_passed: bool = False) -> None:
        assert n % 2 == 0
        self.has_passed = has_passed
        self.n = n

        self.previous_move = None
        if game_state is None:
            self.board = [[-1] * n for _ in range(n)]
            self.board[n // 2][n // 2] = 0
            self.board[n // 2][n // 2 - 1] = 1
            self.board[n // 2 - 1][n // 2] = 1
            self.board[n // 2 - 1][n // 2 - 1] = 0
            self.turn = True
        else:
            self.board = copy.deepcopy(game_state.board)
            self.turn = game_state.turn
            self.n = game_state.n
            self.has_passed = game_state.has_passed
            self.previous_move = game_state.previous_move

    def vector_representation(self) -> List[float]:
        """Return the flattened board"""
        vector = []
        for row in self.board:
            vector.extend(row)
        return vector

    def is_legal(self, move: Tuple[int, int], direction: Tuple[int, int] = (0, 0)) -> bool:
        """Return whether the next move is legal from the game state in self.

        direction is a tuple representing which direction to check.
        If this is (0, 0) all directions are checked.

        Preconditions:
            - 0 <= move[0] <= self.n
            - 0 <= move[1] <= self.n
            - direction in {(0, 0), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1), (0, 1), (0, -1)}
        """
        if self.board[move[0]][move[1]] != -1:
            return False

        if direction == (0, 0):
            possible_directions = {(1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1), (0, 1), (0, -1)}
            return any(self.is_legal(move, new_direction) for new_direction in possible_directions)

        if self.turn:
            move_piece = 1
        else:
            move_piece = 0

        check = (move[0] + direction[0], move[1] + direction[1])

        # There has to exist a piece to capture for a move to be legal
        if not self.in_bounds(check) or self.board[check[0]][check[1]] != 1 - move_piece:
            return False

        check = (move[0] + direction[0], move[1] + direction[1])

        # Checks if there is no empty space before we reach our piece again
        while self.in_bounds(check) and self.board[check[0]][check[1]] == 1 - move_piece:
            check = (check[0] + direction[0], check[1] + direction[1])

        # Return whether we eventually reach our piece again, that is whether our opponents pieces are
        # sandwiched between ours, or if one side is surrounded with our piece and the other with the
        # edge of the board or empty space.
        if not self.in_bounds(check) or self.board[check[0]][check[1]] == -1:
            return False
        return True

    def in_bounds(self, move: Tuple[int, int]):
        """Check if the move is in the bounds of the game.
        A helper function"""
        return 0 <= move[0] <= self.n - 1 and 0 <= move[1] <= self.n - 1

    def make_move(self, move: Optional[Tuple[int, int]], check_legal: bool = True) -> bool:
        """Play move. Returns False if move is not legal and True otherwise.
        move is None when a pass is played.

        check_legal can be set to False to save time.

        Preconditions:
            - 0 <= move[0] <= self.n
            - 0 <= move[1] <= self.n
        """
        if move is None:
            if self.has_passed:
                return False
            self.has_passed = True
            self.turn = not self.turn

            return True

        if not check_legal or self.is_legal(move):
            self.previous_move = move

            possible_directions = {(1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1), (0, 1), (0, -1)}

            for direction in possible_directions:
                if self.is_legal(move, direction):
                    self.reverse_direction(move, direction)

            if self.turn:
                self.board[move[0]][move[1]] = 1
            else:
                self.board[move[0]][move[1]] = 0

            self.turn = not self.turn
            self.has_passed = False
            return True
        else:
            return False

    def reverse_direction(self, start: Tuple[int, int], direction: Tuple[int, int]) -> None:
        """Reverse all pieces sandwiched between two white pieces if piece is 0, and two
        black pieces if piece is 1, in the direction of direction.

        Preconditions:
            - self.is_legal(start, direction)
        """
        if self.turn:
            piece = 1
        else:
            piece = 0

        check = (start[0] + direction[0], start[1] + direction[1])
        while self.board[check[0]][check[1]] != piece:
            self.board[check[0]][check[1]] = piece
            check = (check[0] + direction[0], check[1] + direction[1])

    def evaluate_position(self, heuristic_type: int = 0) -> float:
        """Return an evaluation of the current position.

        heuristic_type 0 is 1 is returned if X wins and -1 is returned if O wins. 0 is returned otherwise.
        heuristic_type 1 returns the number of black pieces subtracted from the number of white pieces,
            normalised by (1 / self.n).
        """
        if heuristic_type == 0:
            winner = self.winner()
            if winner == (True, True):
                return 1
            elif winner == (True, False):
                return -1
            return 0
        if heuristic_type == 1:
            num_black = 0
            num_white = 0
            for row in range(self.n):
                for column in range(self.n):
                    piece = self.board[row][column]
                    if piece == 0:
                        num_white += 1
                    elif piece == 1:
                        num_black += 1

            return (num_black - num_white) / self.n

    def legal_moves(self) -> list[GameState]:
        """Return all legal moves from this position"""
        possible_moves = []
        for i in range(self.n):
            for j in range(self.n):
                if self.is_legal((i, j)):
                    new_game = ReversiGameState(self.n, self)
                    new_game.make_move((i, j), False)
                    possible_moves.append(new_game)

        # You can only pass when you cannot play any other moves.
        if not self.has_passed and possible_moves == []:
            new_game = ReversiGameState(self.n, self)
            new_game.make_move(None, False)
            possible_moves.append(new_game)
        return possible_moves

    def winner(self) -> Optional[Tuple[bool, bool]]:
        """Return (True, True) if X won, (True, False) if O won,
        (False, False) if there is a tie, and None if the game is not over."""
        if self.legal_moves() != []:
            return None

        net_black = self.evaluate_position(1)
        if net_black > 0:
            return (True, True)
        elif net_black < 0:
            return (True, False)
        return (False, False)

    def board_object(self, x, y) -> str:
        """Return a string representing the piece
        at the location (x, y) on the board
        """
        piece = self.board[x][y]
        if piece == 1:
            return 'B'
        elif piece == 0:
            return 'W'
        else:
            return ''

    def equal(self, game_state: ReversiGameState) -> bool:
        """Return whether self is equal to game_state"""
        return self.board == game_state.board and self.has_passed == game_state.has_passed

    def __str__(self) -> str:
        """A unique string representation of the board for memoization and debugging."""
        state_string = ""
        for row in self.board:
            for piece in row:
                if piece == -1:
                    state_string += " - "
                elif piece == 0:
                    state_string += " W "
                else:
                    state_string += " B "
            state_string += "\n"
        return state_string

    def display(self, screen: pygame.display) -> None:
        """Display the current Reversi Board on screen"""
        w, h = screen.get_size()
        background_color = (222, 184, 135)
        screen.fill(background_color)

        # Draw the lines on the board
        for i in range(1, self.n):
            pygame.draw.line(screen, (0, 0, 0), (0, h * i // self.n), (w, h * i // self.n))
            pygame.draw.line(screen, (0, 0, 0), (w * i // self.n, 0), (w * i // self.n, h))

        # Draw the markers
        for x in range(self.n):
            for y in range(self.n):
                if self.board[x][y] == 1:
                    color = (0, 0, 0)
                elif self.board[x][y] == 0:
                    color = (255, 255, 255)
                else:
                    color = background_color

                pygame.draw.circle(
                    screen,
                    color,
                    (
                        (y + 0.5) * (w // self.n),
                        (x + 0.5) * (h // self.n)
                    ),
                    h // (3 * self.n)
                )
        pygame.display.update()

    def get_human_input(self, screen: pygame.display, click_loc: Optional[Tuple[int, int]]) -> Optional[GameState]:
        """Return the game state after a valid move has been inputted by the user"""
        if click_loc is None:
            return None
        w, h = screen.get_size()
        position = ((self.n * click_loc[1]) // h, (self.n * click_loc[0]) // w)

        new_game = ReversiGameState(self.n, self)
        if new_game.make_move(position, False):
            return new_game
        return None

    def copy(self) -> ReversiGameState:
        """Return a copy of self"""
        return ReversiGameState(game_state=self)
