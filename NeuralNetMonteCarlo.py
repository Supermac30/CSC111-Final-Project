"""Holds the Monte Carlo Search Tree and Neural Network"""

from Game import GameState, Game, Player


class MonteCarloSimulation(Player):
    """A player that makes decisions by simulating games and a monte carlo tree search"""
    def choose_move(self, game: Game, opponent_move: GameState) -> GameState:
        """Choose a good move by simulating random games"""
        pass


class MonteCarloNeuralNetwork(Player):
    """A player that makes decisions by a Neural Network and a monte carlo tree search"""

    def choose_move(self, game: Game, opponent_move: GameState) -> GameState:
        """Choose a good move by using evaluations of the current game state from a Neural Network"""
        pass
