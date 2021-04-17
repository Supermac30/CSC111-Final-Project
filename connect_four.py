"""Holds the Connect Four Game

This file is Copyright (c) 2020 Mark Bedaywi
"""
from __future__ import annotations
from typing import Optional, Tuple, List
import copy

import pygame
from game import GameState


class ConnectFourGameState(GameState):
    """Stores the game state of a TicTacToe game

    Instance Attributes:
        - n: The dimension of the board. Must be at least 4.
        - board: A 2D nxn list storing the object in each position in the game.
            A 1 is placed if player 1's piece is in the location,
            0 if it is player 2's piece and -1 if it is empty.
        - turn: Stores the turn of the player. This is true if it is X's turn and False otherwise.
        - previous_move: Stores the previous move made. This is None if no move has been made yet.
    """
    n: int
    board: list[list[int]]
    turn: bool
    previous_move: Optional[int]

    def __init__(self, n: int = 6, game_state: Optional[ConnectFourGameState] = None) -> None:
        assert n >= 4

        self.previous_move = None
        if game_state is None:
            self.board = [[-1] * n for _ in range(n)]
            self.turn = True
        else:
            self.board = copy.deepcopy(game_state.board)
            self.turn = game_state.turn
            self.n = game_state.n
            self.previous_move = game_state.previous_move

        self.n = n

    def vector_representation(self) -> List[float]:
        """Return the flattened board"""
        vector = []
        for row in self.board:
            vector.extend(row)
        return vector

    def is_legal(self, move: int) -> bool:
        """Return whether the next move is legal from the game state in self

        Preconditions:
            - 0 <= move[0] <= 3
            - 0 <= move[1] <= 3
        """
        return self.board[0][move] == -1

    def make_move(self, move: int, check_legal: bool = True) -> bool:
        """Play move. Returns False if move is not legal and True otherwise.

        Preconditions:
            - 0 <= move <= self.n
        """
        if not check_legal or self.is_legal(move):
            self.previous_move = move
            if self.turn:
                piece = 1
            else:
                piece = 0

            placed_piece = False
            row = 0
            while not placed_piece and row < self.n - 1:
                row += 1
                if self.board[row][move] != -1:
                    self.board[row - 1][move] = piece
                    placed_piece = True

            if not placed_piece:
                self.board[-1][move] = piece

            if row == self.n:
                self.board[-1][move] = piece

            self.turn = not self.turn
            return True
        else:
            return False

    def evaluate_position(self, heuristic_type: int = 0) -> float:
        """Return an evaluation of the current position.
        There is only the default heuristic for Connect 4:
        1 is returned if X wins and -1 is returned if O wins. 0 is returned otherwise.
        """
        winner = self.winner()
        if winner == (True, True):
            return 1
        elif winner == (True, False):
            return -1
        return 0

    def legal_moves(self) -> list[GameState]:
        """Return all legal moves from this position"""

        # Checks if the game is over
        if self.winner() is not None:
            return []

        possible_moves = []
        for i in range(self.n):
            if self.is_legal(i):
                new_game = ConnectFourGameState(self.n, self)
                new_game.make_move(i, False)
                possible_moves.append(new_game)
        return possible_moves

    def winner(self) -> Optional[Tuple[bool, bool]]:
        """Return (True, True) if Red won, (True, False) if Yellow won,
        (False, False) if there is a tie, and None if the game is not over."""

        # Check Horizontals
        for row in range(self.n):
            for column in range(self.n - 3):
                if all(self.board[row][column + i] == 1 for i in range(4)):
                    return (True, True)
                if all(self.board[row][column + i] == 0 for i in range(4)):
                    return (True, False)

        # Check Verticals
        for column in range(self.n):
            for row in range(self.n - 3):
                if all(self.board[row + i][column] == 1 for i in range(4)):
                    return (True, True)
                if all(self.board[row + i][column] == 0 for i in range(4)):
                    return (True, False)

        # Check Decreasing Diagonals
        for column in range(self.n - 3):
            for row in range(self.n - 3):
                if all(self.board[row + i][column + i] == 1 for i in range(4)):
                    return (True, True)
                if all(self.board[row + i][column + i] == 0 for i in range(4)):
                    return (True, False)

        # Check Increasing Diagonals
        for column in range(3, self.n):
            for row in range(self.n - 3):
                if all(self.board[row + i][column - i] == 1 for i in range(4)):
                    return (True, True)
                if all(self.board[row + i][column - i] == 0 for i in range(4)):
                    return (True, False)

        is_over = all(
            self.board[i][j] != -1
            for i in range(self.n)
            for j in range(self.n)
        )

        if is_over:
            return (False, False)
        else:
            return None

    def board_object(self, x: int, y: int) -> str:
        """Return a string representing the piece
        at the location (x, y) on the board
        """
        piece = self.board[x][y]
        if piece == 1:
            return 'R'
        elif piece == 0:
            return 'Y'
        else:
            return ''

    def equal(self, game_state: ConnectFourGameState) -> bool:
        """Return whether self is equal to game_state"""
        return self.board == game_state.board

    def __str__(self) -> str:
        """A unique string representation of the board for memoization and debugging."""
        state_string = ""
        for row in self.board:
            for piece in row:
                if piece == -1:
                    state_string += " - "
                elif piece == 0:
                    state_string += " Y "
                else:
                    state_string += " R "
            state_string += "\n"
        return state_string

    def display(self, screen: pygame.display) -> None:
        """Display the current Connect Four Board on screen"""
        w, h = screen.get_size()
        screen.fill((0, 0, 255))

        # Draw the lines on the board
        for i in range(1, self.n):
            pygame.draw.line(screen, (0, 0, 0), (0, h * i // self.n), (w, h * i // self.n))
            pygame.draw.line(screen, (0, 0, 0), (w * i // self.n, 0), (w * i // self.n, h))

        # Draw the markers
        for x in range(self.n):
            for y in range(self.n):
                if self.board[x][y] == 1:
                    color = (255, 0, 0)
                elif self.board[x][y] == 0:
                    color = (255, 255, 0)
                else:
                    color = (255, 255, 255)

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

    def get_human_input(self, screen: pygame.display,
                        click_loc: Optional[Tuple[int, int]]) -> Optional[GameState]:
        """Return the game state after a valid move has been inputted by the user"""
        if click_loc is None:
            return None
        w = screen.get_size()[0]
        position = (self.n * click_loc[0]) // w

        new_game = ConnectFourGameState(self.n, self)
        if new_game.make_move(position, True):
            return new_game
        return None

    def copy(self) -> ConnectFourGameState:
        """Return a copy of self"""
        return ConnectFourGameState(self.n, self)


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        'max-line-length': 100,
        'disable': ['E1136']
    })
