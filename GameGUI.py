"""Holds the GUI where games can be played"""
from typing import Tuple

import Game
import Player


def display_game(history: list[Game.GameState], screen_size: Tuple[int, int] = (500, 500)):
    """Builds a GUI to display the sequence of game states in history.

    Precondition:
        - len(history) != 0
    """
    import pygame
    pygame.init()
    screen = pygame.display.set_mode(screen_size)
    position = 0
    complete = False

    while not complete:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                complete = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    position = max(0, position - 1)
                elif event.key == pygame.K_RIGHT:
                    position = min(len(history) - 1, position + 1)
        screen.fill((255, 255, 255))
        history[position].display(screen)

        pygame.display.flip()

    pygame.quit()


def start() -> list[Game.GameState]:
    """A test function"""
    import time
    import TicTacToe
    b = time.time()
    game_tree1 = Player.MinimaxGameTree(TicTacToe.TicTacToeGameState())
    game_tree2 = Player.MinimaxGameTree(TicTacToe.TicTacToeGameState())
    player1 = Player.MinimaxPlayer(1, game_tree1)
    player2 = Player.MinimaxPlayer(2, game_tree2, 10)

    game = TicTacToe.TicTacToe(player1, player2, TicTacToe.TicTacToeGameState())
    x = game.play_game()[1]
    print(time.time() - b)
    return x

display_game(start())
