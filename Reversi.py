"""Holds the Reversi Game"""
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
        - game_type: Holds the type of game.
        - previous_move: Stores the previous move made. This is None if no move has been made yet.
    """
    n: int
    board: list[list[int]]
    turn: bool
    game_type: Type[Game]
    previous_move: Optional[Tuple[int, int]]

    def __init__(self, n: int = 6, game_state: Optional[ReversiGameState] = None) -> None:
        assert n % 2 == 0

        self.n = n

        self.game_type = Reversi
        self.previous_move = None
        if game_state is None:
            self.board = [[-1] * n for _ in range(n)]
            self.board[n // 2][n // 2] = 1
            self.board[n // 2][n // 2 - 1] = 0
            self.board[n // 2 - 1][n // 2] = 0
            self.board[n // 2 - 1][n // 2 - 1] = 1
            self.turn = True
        else:
            self.board = copy.deepcopy(game_state.board)
            self.turn = game_state.turn

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
        # TODO: finish
        pass

    def make_move(self, move: Tuple[int, int], check_legal: bool = True) -> bool:
        """Play move. Returns False if move is not legal and True otherwise.

        check_legal can be set to False to save time.

        Preconditions:
            - 0 <= move[0] <= self.n
            - 0 <= move[1] <= self.n
        """
        if not check_legal and self.is_legal(move):
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

        heuristic_type 0 is 1 is returned if X wins and -1 is returned if O wins. 0 is returned otherwise.
        heuristic_type 1 returns the number of white pieces subtracted from the number of white pieces,
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
                    if piece == 1:
                        num_white += 1
                    elif piece == 0:
                        num_black += 1

            return (num_black - num_white) / self.n

    def legal_moves(self) -> list[GameState]:
        """Return all legal moves from this position"""

        # Checks if the game is over
        if self.winner() is not None:
            return []

        possible_moves = []
        for i in range(3):
            for j in range(3):
                if self.is_legal((i, j)):
                    new_game = ReversiGameState(self.n, self)
                    new_game.make_move((i, j), False)
                    possible_moves.append(new_game)
        return possible_moves

    def winner(self) -> Optional[Tuple[bool, bool]]:
        """Return (True, True) if X won, (True, False) if O won,
        (False, False) if there is a tie, and None if the game is not over."""
        # TODO

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

    def equal(self, game_state: ReversiGameState) -> bool:
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
                    state_string += " W "
                else:
                    state_string += " B "
            state_string += "\n"
        return state_string

    def display(self, screen: pygame.display) -> None:
        """Display the current TicTacToe Board on screen"""
        w, h = screen.get_size()

        # TODO: fix

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

    def copy(self) -> ReversiGameState:
        """Return a copy of self"""
        return ReversiGameState(self)


class Reversi(Game):
    """A subclass of Game implementing Reversi.

    Instance Attributes:
        - player1: Stores the Player object representing the player playing as 'X'.
        - player2: Stores the Player object representing the player playing as 'O'.
    """
    # Private Instance Attributes
    #   - game_state: Stores the current game state
    _game_state: ReversiGameState

    def copy(self) -> Reversi:
        """Return a copy of self"""
        return Reversi(self.player1.copy(), self.player2.copy(), self._game_state.copy())
