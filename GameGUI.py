"""Holds the GUI where games can be played

This file is Copyright (c) 2020 Mark Bedaywi
"""
from typing import Tuple

import Player
import Game
import TicTacToe
import ConnectFour
import Reversi
import MonteCarloSimulation


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
    b = time.time()

    start_state = TicTacToe.TicTacToeGameState()

    player2 = MonteCarloSimulation.MonteCarloSimulationPlayer(start_state.copy())
    player1 = Player.MinimaxPlayer(start_state.copy())

    game = TicTacToe.TicTacToe(player1, player2, start_state.copy())
    x = game.play_game(True)
    print(x[0])
    print(time.time() - b)
    return x[1]


if __name__ == "__main__":
    display_game(start())
