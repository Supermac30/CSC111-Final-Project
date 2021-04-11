""" Holds the main function

This file is Copyright (c) 2020 Mark Bedaywi
"""
import Game
import Player

"""
    Possible players:
        - Random
        - Variations of Minimax that all search for a certain time
            - Minimax with no optimisations
            - Minimax with alpha-beta pruning
            - Minimax with memoization
            - Minimax with both memoization and alpha-beta pruning
        - Variations of Monte Carlo Tree search
            - Using a Neural Network
            - Using a random simulation
"""

# TODO: Create a GUI for TicTacToe to test algorithms
# TODO: Create a basic minimax player.
# TODO: Implement 9 x 9 Go
# TODO: Implement a Monte Carlo Search Tree
# TODO: Implement a Neural Network to rank positions


def start() -> Game.Game:
    """A test function"""
    import TicTacToe
    import pygame
    game_tree1 = Game.GameTree(TicTacToe.TicTacToeGameState())
    game_tree2 = Game.GameTree(TicTacToe.TicTacToeGameState())
    player1 = Player.RandomPlayer(1, game_tree1)
    player2 = Player.RandomPlayer(2, game_tree2)

    screen = pygame.display.set_mode((500, 500))

    game = TicTacToe.TicTacToe(player1, player2, TicTacToe.TicTacToeGameState(), screen)
    return game
