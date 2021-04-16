"""Holds the TicTacToe Game

This file is Copyright (c) 2020 Mark Bedaywi
"""
from __future__ import annotations
from typing import Optional, Tuple, List
import copy

import pygame

from game import GameState


class TicTacToeGameState(GameState):
    """Stores the game state of a TicTacToe game

    Instance Attributes:
        - board: A 2D 3x3 list storing the object in each position in the game.
            A 1 is placed if 'X' is in the location, 0 if it is a 'O' and -1 if it is empty.
        - turn: Stores the turn of the player. This is true if it is X's turn and False otherwise.
        - previous_move: Stores the previous move made. This is None if no move has been made yet.
    """
    board: list[list[int]]
    turn: bool
    previous_move: Optional[Tuple[int, int]]

    def __init__(self, game_state: Optional[TicTacToeGameState] = None) -> None:
        self.previous_move = None
        if game_state is None:
            self.board = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
            self.turn = True
        else:
            self.board = copy.deepcopy(game_state.board)
            self.turn = game_state.turn
            self.previous_move = game_state.previous_move

    def vector_representation(self) -> List[float]:
        """Return the flattened board"""
        vector = []
        for row in self.board:
            vector.extend(row)
        return vector

    def is_legal(self, move: Tuple[int, int]) -> bool:
        """Return whether the next move is legal from the game state in self

        Preconditions:
            - 0 <= move[0] <= 3
            - 0 <= move[1] <= 3
        """
        return self.board[move[0]][move[1]] == -1

    def make_move(self, move: Tuple[int, int], check_legal: bool = True) -> bool:
        """Play move. Returns False if move is not legal and True otherwise.

        check_legal can be made false to save time

        Preconditions:
            - 0 <= move[0] <= 3
            - 0 <= move[1] <= 3
        """
        if not check_legal or self.is_legal(move):
            self.previous_move = move
            if self.turn:
                self.board[move[0]][move[1]] = 1
            else:
                self.board[move[0]][move[1]] = 0
            self.turn = not self.turn
            return True
        else:
            return False

    def evaluate_position(self, heuristic_type: int = 0) -> float:
        """Return an evaluation of the current position.
        There is only the default heuristic for TicTacToe:
        1 is returned if X wins and -1 is returned if O wins.
        0 is returned otherwise.
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
        for i in range(3):
            for j in range(3):
                if self.is_legal((i, j)):
                    new_game = TicTacToeGameState(self)
                    new_game.make_move((i, j), False)
                    possible_moves.append(new_game)
        return possible_moves

    def winner(self) -> Optional[Tuple[bool, bool]]:
        """Return (True, True) if X won, (True, False) if O won,
        (False, False) if there is a tie, and None if the game is not over."""
        # Checks vertical lines
        for i in range(3):
            if self.board[0][i] == self.board[1][i] == self.board[2][i]:
                if self.board[0][i] == 1:
                    return (True, True)
                elif self.board[0][i] == 0:
                    return (True, False)

        # Checks horizontal lines
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2]:
                if self.board[i][0] == 1:
                    return (True, True)
                elif self.board[i][0] == 0:
                    return (True, False)

        # Checks the forward diagonal
        if self.board[0][0] == self.board[1][1] == self.board[2][2]:
            if self.board[0][0] == 1:
                return (True, True)
            elif self.board[0][0] == 0:
                return (True, False)

        # Checks the backwards diagonal
        if self.board[0][2] == self.board[1][1] == self.board[2][0]:
            if self.board[0][2] == 1:
                return (True, True)
            elif self.board[0][2] == 0:
                return (True, False)

        is_over = all(
            self.board[row][column] != -1
            for row in range(3)
            for column in range(3)
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
            return 'X'
        elif piece == 0:
            return 'O'
        else:
            return ''

    def equal(self, game_state: TicTacToeGameState) -> bool:
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
                    state_string += " O "
                else:
                    state_string += " X "
            state_string += "\n"
        return state_string

    def display(self, screen: pygame.display) -> None:
        """Display the current TicTacToe Board on screen"""
        w, h = screen.get_size()
        screen.fill((255, 255, 255))

        # Draw the lines on the board
        pygame.draw.line(screen, (0, 0, 0), (0, h // 3), (w, h // 3))
        pygame.draw.line(screen, (0, 0, 0), (0, 2 * h // 3), (w, 2 * h // 3))
        pygame.draw.line(screen, (0, 0, 0), (w // 3, 0), (w // 3, h))
        pygame.draw.line(screen, (0, 0, 0), (2 * w // 3, 0), (2 * w // 3, h))

        # Draw the markers
        font = pygame.font.SysFont('Calibri', 100)
        for x in range(3):
            for y in range(3):
                piece = font.render(
                    self.board_object(x, y),
                    True,
                    (0, 0, 0)
                )
                screen.blit(
                    piece,
                    (
                        (y + 0.3) * (w // 3),
                        (x + 0.3) * (h // 3)
                    )
                )
        pygame.display.update()

    def get_human_input(self, screen: pygame.display,
                        click_loc: Optional[Tuple[int, int]]) -> Optional[GameState]:
        """Return the game state after a valid move has been inputted by the user"""
        if click_loc is None:
            return None
        w, h = screen.get_size()
        position = ((3 * click_loc[1]) // w, (3 * click_loc[0]) // h)

        new_game = TicTacToeGameState(self)
        if new_game.make_move(position, False):
            return new_game
        return None

    def copy(self) -> TicTacToeGameState:
        """Return a copy of self"""
        return TicTacToeGameState(self)


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'max-line-length': 100,
        'disable': ['E1136']
    })
