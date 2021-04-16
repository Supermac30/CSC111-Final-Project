"""Holds the Monte Carlo Search Tree that uses a Neural Network

This file is Copyright (c) 2020 Mark Bedaywi
"""
from __future__ import annotations

import copy
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

        - neural_network: Holds the MLPClassifier that takes in a state and returns two values.
            The first is the predicted value of moving into the state. This is used in the MCST to
                update the value of a state.
            The second is the probability that a state should be explored. This is used in the MCST to
                choose which nodes to explore.
    """

    root: GameState
    value: Optional[float]
    children: list[MonteCarloNeuralNetwork]
    repeat: int
    exploration_parameter: float
    visits: int

    neural_network: MLPClassifier

    def __init__(self, start_state: GameState, neural_network: MLPClassifier,
                 repeat: int = 40, exploration_parameter: float = 1.4142, value: float = 0) -> None:
        super().__init__(start_state, repeat=repeat,
                         exploration_parameter=exploration_parameter, value=value)
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

        exploitation_value = self.value / self.visits
        exploration_value = self.exploration_parameter * (math.sqrt(visits_parent) / (1 + self.visits))

        return exploitation_value + exploration_value

    def move_value(self) -> float:
        """Estimate the value of the root using the neural network.

        Returns the true value if self is terminal
        """
        # Return the true value if the state is terminal
        winner = self.root.winner()
        if self.root.winner() is not None:
            if winner[0]:  # If there was not a tie
                if self.root.turn != winner[1]:
                    return 1
                return 0
            return 0.5

        # Return the value predicted by the neural network
        player_1_reward = self.neural_network.predict([self.root.vector_representation()])[0]
        # Normalises the categories into values between 0 and 1
        player_1_reward = (player_1_reward + 1) / 2

        if not self.root.turn:
            return player_1_reward
        return 1 - player_1_reward

    def copy(self) -> MonteCarloNeuralNetwork:
        """Return a copy of self"""
        new_tree = MonteCarloNeuralNetwork(
            self.root.copy(),
            self.neural_network,
            self.repeat,
            self.exploration_parameter,
            self.value
        )
        new_tree.children = [child.copy() for child in self.children]
        return new_tree


class MonteCarloNeuralNetworkPlayer(Player):
    """A player that chooses the optimal move using a Monte Carlo search tree with simulation

    Instance Attributes:
        - game_tree: Holds the GameTree object the player uses to make decisions
    """
    game_tree: MonteCarloNeuralNetwork

    def __init__(self, start_state: GameState, neural_network: MLPClassifier,
                 game_tree: MonteCarloNeuralNetwork = None, repeat: int = 50) -> None:
        if game_tree is not None:
            self.game_tree = game_tree
        else:
            self.game_tree = MonteCarloNeuralNetwork(start_state, neural_network, repeat=repeat)

    def choose_move(self) -> GameState:
        """Return the optimal move from the game state in self.game_tree.root

        Assumes the game is not over, that is, assumes there are possible
        legal moves from this position
        """
        self.game_tree.find_value()

        best_move = self.game_tree.children[0]
        best_average_value = -float("inf")
        for move in self.game_tree.children:
            if move.visits == 0:
                continue
            average_value = move.value / move.visits

            if average_value > best_average_value:
                best_move = move
                best_average_value = best_move.value / best_move.visits

        return best_move.root

    def copy(self) -> MonteCarloNeuralNetworkPlayer:
        """Return a copy of self"""
        return MonteCarloNeuralNetworkPlayer(self.game_tree.root,
                                             self.game_tree.neural_network, self.game_tree.copy())


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
        return NeuralNetworkPlayer(self.game_tree.root.copy(), self.neural_network,
                                   self.is_player1, self.game_tree.copy())


def train_neural_network(game_state: Type[GameState], hidden_layer: Union[int, Tuple],
                         repeat: int = 10, num_games: int = 10, neural_net: MLPClassifier = None) -> MLPClassifier:
    """Trains a neural network to play TicTacToe.

    The AI plays against itself num_games times, continuously updating and improving.
    """
    if neural_net is None:
        neural_net = MLPClassifier(hidden_layer_sizes=hidden_layer, max_iter=2000)

        # initializes the neural network arbitrarily
        initial_x = [game_state().vector_representation(), game_state().vector_representation(),
                     game_state().vector_representation()]
        initial_y = [[-1], [0], [1]]
        neural_net.fit(initial_x, initial_y)

    training = ([], [])
    for i in range(num_games):
        training, neural_net = update_neural_network(game_state, neural_net, repeat, training)
        print(i)

    return neural_net


def update_neural_network(game_state: Type[GameState], neural_net: MLPClassifier, repeat: int,
                          training: Tuple[list[list], list[float]]) \
        -> Tuple[Tuple[list[list], list[float]], MLPClassifier]:
    """A helper function that has neural_net play a game against itself, then learn.

    Returns a tuple where the first element is the training data, and the second
    is the new neural network.
    """
    # set up the game

    player1 = MonteCarloNeuralNetworkPlayer(game_state(), neural_net, True, repeat=repeat)
    player2 = MonteCarloNeuralNetworkPlayer(game_state(), neural_net, False, repeat=repeat)

    set_up_game = Game(player1, player2)

    # play the game
    winner, history = set_up_game.play_game(False)
    print(winner)

    # train the neural network

    if not winner[0]:
        state_value = 0
    elif winner[1]:
        state_value = 1
    else:
        state_value = -1

    x = training[0]
    y = training[1]

    x.extend([state.vector_representation() for state in history])
    y.extend([state_value] * len(history))

    old_neural_net = copy.deepcopy(neural_net)
    neural_net.fit(x, y)

    if not is_better(game_state, neural_net, old_neural_net):
        return (x, y), old_neural_net
    print("new is better")
    return (x, y), neural_net


def is_better(game_state: Type[GameState], neural_net_1: MLPClassifier,
              neural_net_2: MLPClassifier, num_games: int = 2) -> bool:
    """Return whether neural_net1 beats neural_net2 more often"""
    player1 = MonteCarloNeuralNetworkPlayer(game_state(), neural_net_1, True)
    player2 = MonteCarloNeuralNetworkPlayer(game_state(), neural_net_2, False)

    set_up_game = Game(player1, player2)
    num_wins_1 = set_up_game.play_games(num_games)[0]
    print(num_wins_1)
    if num_wins_1 == 0:
        return False

    player1 = MonteCarloNeuralNetworkPlayer(game_state(), neural_net_2, True)
    player2 = MonteCarloNeuralNetworkPlayer(game_state(), neural_net_1, False)

    set_up_game = Game(player1, player2)
    num_wins_2 = set_up_game.play_games(num_games)[1]
    print(num_wins_2)

    # Return whether neural_net1 won a majority of the 2 * num_games games
    return num_wins_1 + num_wins_2 > num_games


def test_neural_network(game_state: Type[GameState], neural_network: MLPClassifier, is_player1: bool) -> None:
    import GameGUI
    import Player

    player1 = MonteCarloNeuralNetworkPlayer(game_state(), neural_network, is_player1)
    player2 = Player.MinimaxPlayer(game_state(), depth=1)

    if is_player1:
        set_up_game = Game(player1, player2)
    else:
        set_up_game = Game(player2, player1)
    GameGUI.display_game(set_up_game.play_with_human(not is_player1)[1])


def save_neural_network(neural_network: MLPClassifier, file_name: str):
    """Save the trained neural network in the file file_name.

    This uses the library pickle.
    """
    import pickle
    pickle.dump(neural_network, open(file_name, 'wb'))


def load_neural_network(file_name: str) -> MLPClassifier:
    """Return the trained neural network in the file file_name

    This uses the library pickle
    """
    import pickle
    return pickle.load(open(file_name, 'rb'))


if __name__ == "__main__":
    value = True
    if value:
        # import TicTacToe
        # brain = train_neural_network(TicTacToe.TicTacToeGameState, (6, 3), repeat=50, num_games=100)
        # save_neural_network(brain, "data/neural_networks/TicTacToeNeuralNetwork.txt")

        # import ConnectFour
        # brain = train_neural_network(ConnectFour.ConnectFourGameState, (6, 6), repeat=100, num_games=300)
        # save_neural_network(brain, "data/neural_networks/ConnectFourNeuralNetwork.txt")

        import Reversi
        old_brain = load_neural_network("data/neural_networks/ReversiNeuralNetwork.txt")
        brain = train_neural_network(Reversi.ReversiGameState, (8, 8), repeat=200, num_games=100, neural_net=old_brain)
        save_neural_network(brain, "data/neural_networks/ReversiNeuralNetwork.txt")

    else:
        import Reversi
        import GameGUI
        from Player import RandomPlayer
        from MonteCarloSimulation import MonteCarloSimulationPlayer
        start_state = Reversi.ReversiGameState()
        brain = load_neural_network("data/neural_networks/ReversiNeuralNetwork.txt")
        player1 = MonteCarloNeuralNetworkPlayer(start_state.copy(), brain, True)
        player1p = NeuralNetworkPlayer(start_state.copy(), brain, False)
        player2 = RandomPlayer(start_state.copy(),)

        game = Game(player1, player2)
        print(GameGUI.display_game(game.play_game()[1]))
