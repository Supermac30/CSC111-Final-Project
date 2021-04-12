"""Holds the TicTacToe Game"""
from __future__ import annotations
from typing import Optional, Tuple, Type

from Game import Game, GameState
import pygame
import copy


class TicTacToeGameState(GameState):
    """Stores the game state of a TicTacToe game

    Instance Attributes:
        - game_state: A 2D 3x3 list storing the object in each position in the game.
            A 1 is placed if 'X' is in the location, 0 if it is a 'O' and -1 if it is empty.
        - turn: Stores the turn of the player. This is true if it is X's turn and False otherwise.
        - game_type: Holds the type of game.
        - previous_move: Stores the previous move made. This is None if no move has been made yet.
    """
    board: list[list[int]]
    turn: bool
    game_type: Type[Game]
    previous_move: Optional[Tuple[int, int]]

    def __init__(self, game_state: Optional[TicTacToeGameState] = None) -> None:
        self.game_type = TicTacToe
        self.previous_move = None
        if game_state is None:
            self.board = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
            self.turn = True
        else:
            self.board = copy.deepcopy(game_state.board)
            self.turn = game_state.turn

    def is_legal(self, move: Tuple[int, int]) -> bool:
        """Return whether the next move is legal from the game state in self

        Preconditions:
            - 0 <= move[0] <= 3
            - 0 <= move[1] <= 3
        """
        return self.board[move[0]][move[1]] == -1

    def make_move(self, move: Tuple[int, int]) -> bool:
        """Play move. Returns False if move is not legal and True otherwise.

        Preconditions:
            - 0 <= move[0] <= 3
            - 0 <= move[1] <= 3
        """
        if self.is_legal(move):
            self.previous_move = move
            if self.turn:
                self.board[move[0]][move[1]] = 1
            else:
                self.board[move[0]][move[1]] = 0
            self.turn = not self.turn
            return True
        else:
            return False

    def change_state(self, new_state: GameState) -> bool:
        """Change game state to new_state, making the previous move.
         Returns False if new_state is not legal from this position.

        Preconditions:
            - 0 <= move[0] <= 3
            - 0 <= move[1] <= 3
        """
        move = new_state.previous_move
        return self.make_move(move)

    def evaluate_position(self, heuristic_type: int = 0) -> float:
        """Return an evaluation of the current position.
        There is only the default heuristic for TicTacToe:
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
        for i in range(3):
            for j in range(3):
                if self.is_legal((i, j)):
                    new_game = TicTacToeGameState(self)
                    new_game.make_move((i, j))
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
            self.board[i][j] != -1
            for i in range(3)
            for j in range(3)
        )

        if is_over:
            return (False, False)
        else:
            return None

    def board_object(self, x, y) -> str:
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
                        (y + 0.5) * (w // 3) - 30,
                        (x + 0.5) * (h // 3) - 30
                    )
                )
        pygame.display.update()

    def copy(self) -> TicTacToeGameState:
        """Return a copy of self"""
        return TicTacToeGameState(self)


class TicTacToe(Game):
    """A subclass of Game implementing TicTacToe.

    This is used as a simple testing ground for algorithms

    Instance Attributes:
        - player1: Stores the Player object representing the player playing as 'X'.
        - player2: Stores the Player object representing the player playing as 'O'.
    """
    # Private Instance Attributes
    #   - game_state: Stores the current game state
    #   - _screen: Stores the pygame screen displaying the Game
    _game_state: TicTacToeGameState
    _screen: pygame.display
