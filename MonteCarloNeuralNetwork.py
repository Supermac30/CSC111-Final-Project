"""Holds the Monte Carlo Search Tree that uses a Neural Network

This file is Copyright (c) 2020 Mark Bedaywi
"""
from __future__ import annotations

import math
from typing import Optional, Type, Tuple, Union

from Game import Game, GameState, GameTree, Player, MoveNotLegalError
from MonteCarloSimulation import MonteCarloGameTree

from sklearn.neural_network import MLPClassifier


class MonteCarloNeuralNetwork(MonteCarloGameTree):
    """A player that estimates the value of states by using a Neural network.

    Instance Attributes:
        - root: Holds the GameState in the root of self.
        - value: Holds the value of the root state. This is None if the value has not been calculated yet.
        - children: Holds all subtrees of self connected to the root.
        - repeat: Holds the number of times a Monte Carlo tree search is performed to estimate the value of root.
        - exploration_parameter: Holds a value representing how much the AI should explore rather than exploit.
        - visits: Holds the number of times self has been simulated.
    """

    root: GameState
    value: Optional[float]
    children: list[MonteCarloNeuralNetwork]
    repeat: int
    exploration_parameter: float
    visits: int
    is_player1: bool

    neural_network: MLPClassifier

    def __init__(self, start_state: GameState, neural_network: MLPClassifier, is_player1: bool,
                 repeat: int = 2, exploration_parameter: float = 2, value: float = 0) -> None:
        super().__init__(start_state, is_player1,
                         repeat=repeat, exploration_parameter=exploration_parameter, value=value)
        self.neural_network = neural_network

    def expand_tree(self, state: GameState) -> None:
        """Add all children of state in self, if they are not already there.
        Adds a MinimaxGameTree instead of the generic GameTree

        Assumes that if some child is present, then all possible children are present.
        """
        if state == self.root:
            if self.children == []:
                self.children = [MonteCarloNeuralNetwork(
                    move,
                    self.neural_network,
                    self.is_player1,
                    repeat=self.repeat,
                    exploration_parameter=self.exploration_parameter
                ) for move in self.root.legal_moves()]
        else:
            for child in self.children:
                child.expand_tree(state)

    def make_move(self, state: GameState) -> None:
        """Makes a move, updating root and children
        Updates the value of self.value

        Raises a MoveError if move not in children
        """
        for child in self.children:
            if child.root.previous_move == state.previous_move:
                self.children = child.children
                self.root = state
                self.value = child.value
                self.visits = child.visits

                return

        raise MoveNotLegalError(str(state.previous_move))

    def ucb(self, visits_parent: int) -> float:
        """Use the upper confidence bound to give a value
        representing to what extent a state is worth exploring."""
        exploration_value = self.exploration_parameter \
            * self.move_value()\
            * (math.sqrt(visits_parent) / (1 + self.visits))

        # exploration_value is greater if the position is better for player 1
        if not self.is_player1:
            exploration_value *= -1

        return self.value + exploration_value

    def move_value(self) -> float:
        """Estimate the value of the root using the neural network.

        If an end state is reached, then the actual reward is returned."""
        self.visits += 1

        winner = self.root.winner()
        if winner is not None:
            if not winner[0]:
                return 0
            if winner[1]:
                return 1
            else:
                return -1

        return self.neural_network.predict([self.root.vector_representation()])[0]

    def update_value(self) -> None:
        """Update the value of self by using the values of the children in the backpropagation phase"""
        self.visits = sum([child.visits for child in self.children])
        total_children_values = sum([child.value for child in self.children])
        self.value = (self.value * self.visits + total_children_values) / (len(self.children) + self.visits)

    def copy(self) -> MonteCarloNeuralNetwork:
        """Return a copy of self"""
        return MonteCarloNeuralNetwork(
            self.root.copy(),
            self.neural_network,
            self.is_player1,
            self.repeat,
            self.exploration_parameter,
            self.value
        )


class MonteCarloNeuralNetworkPlayer(Player):
    """A player that chooses the optimal move using a Monte Carlo search tree with simulation

    Instance Attributes:
        - game_tree: Holds the GameTree object the player uses to make decisions
        - is_player1: Holds whether this player is player 1
    """
    game_tree: MonteCarloNeuralNetwork
    is_player1: bool

    def __init__(self, start_state: GameState, neural_network: MLPClassifier,
                 is_player1: bool, game_tree: MonteCarloNeuralNetwork = None, repeat: int = 2) -> None:
        if game_tree is not None:
            self.game_tree = game_tree
        else:
            self.game_tree = MonteCarloNeuralNetwork(start_state, neural_network, is_player1, repeat=repeat)
        self.is_player1 = is_player1

    def choose_move(self) -> GameState:
        """Return the optimal move from the game state in self.game_tree.root

        Assumes the game is not over, that is, assumes there are possible
        legal moves from this position
        """
        best_move = self.game_tree.children[0]
        for move in self.game_tree.children:
            move.find_value()

            if self.is_player1:
                if move.value > best_move.value:
                    best_move = move
            else:
                if move.value < best_move.value:
                    best_move = move

        return best_move.root

    def copy(self) -> MonteCarloNeuralNetworkPlayer:
        """Return a copy of self"""
        return MonteCarloNeuralNetworkPlayer(self.game_tree.root, self.game_tree.neural_network,
                                             self.is_player1, self.game_tree.copy())


class NeuralNetworkPlayer(Player):
    """A Player that uses a trained Neural Network to choose the next moves
    Instance Attributes:
        - game_tree: Holds the GameTree object the player uses to make decisions
        - is_player1: Holds whether this player is player 1
        - neural_network: Holds the trained neural network
    """
    game_tree: GameTree
    neural_network: MLPClassifier
    is_player1: bool

    def __init__(self, start_state: GameState, neural_network: MLPClassifier,
                 is_player1: bool, game_tree: GameTree = None) -> None:
        if game_tree is not None:
            self.game_tree = game_tree
        else:
            self.game_tree = GameTree(start_state)
        self.is_player1 = is_player1
        self.neural_network = neural_network

    def choose_move(self) -> GameState:
        """Choose the optimal move as predicted by the trained neural network"""
        best_move = self.game_tree.children[0]
        for move in self.game_tree.children:
            if self.is_player1:
                if self.state_value(move.root) > self.state_value(best_move.root):
                    best_move = move
            else:
                if self.state_value(move.root) < self.state_value(best_move.root):
                    best_move = move

        return best_move.root

    def state_value(self, state: GameState) -> float:
        """Return the predicted value of the state from the neural network"""
        return self.neural_network.predict([state.vector_representation()])[0]

    def copy(self) -> NeuralNetworkPlayer:
        """Return a copy of self"""
        return NeuralNetworkPlayer(self.game_tree.root, self.neural_network, self.is_player1, self.game_tree)


def train_neural_network(game: Type[Game], game_state: Type[GameState], hidden_layers_sizes: Union[Tuple, int],
                         repeat: int = 2, num_games: int = 10) -> MLPClassifier:
    """Trains a neural network to play TicTacToe.

    The AI plays against itself num_games times, continuously updating and improving.
    """
    neural_net = MLPClassifier(hidden_layer_sizes=hidden_layers_sizes)

    # initializes the neural network arbitrarily
    initial_x = [game_state().vector_representation()]
    initial_y = [0]
    neural_net.fit(initial_x, initial_y)

    for i in range(num_games):
        update_neural_network(game, game_state, neural_net, repeat)
        print(i)

    return neural_net


def update_neural_network(game: Type[Game], game_state: Type[GameState],
                          neural_net: MLPClassifier, repeat: int) -> None:
    """A helper function that has neural_net play a game against itself, then learn"""
    # set up the game

    player1 = MonteCarloNeuralNetworkPlayer(game_state(), neural_net, True, repeat=repeat)
    player2 = MonteCarloNeuralNetworkPlayer(game_state(), neural_net, False, repeat=repeat)

    set_up_game = game(player1, player2, game_state())

    # play the game
    winner, history = set_up_game.play_game(True)
    print(winner)

    # train the neural network
    x = [state.vector_representation() for state in history]

    if not winner[0]:
        state_value = 0
    elif winner[1]:
        state_value = 1
    else:
        state_value = -1

    y = [state_value] * len(x)

    neural_net.fit(x, y)


def test_neural_network(game: Type[Game], game_state: Type[GameState],
                        neural_network: MLPClassifier, is_player1: bool):
    import GameGUI
    import Player

    player1 = MonteCarloNeuralNetworkPlayer(game_state(), neural_network, is_player1)
    player2 = Player.MinimaxPlayer(game_state(), depth=1)

    if is_player1:
        set_up_game = game(player1, player2, game_state())
    else:
        set_up_game = game(player2, player1, game_state())
    GameGUI.display_game(set_up_game.play_game()[1])


import TicTacToe
brain = train_neural_network(TicTacToe.TicTacToe, TicTacToe.TicTacToeGameState, 3, repeat=4, num_games=20)
