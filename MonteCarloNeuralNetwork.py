"""Holds the Monte Carlo Search Tree that uses a Neural Network"""
from __future__ import annotations
from typing import Optional

from Game import GameState, MoveNotLegalError
from MonteCarloSimulation import MonteCarloGameTree

from sklearn.neural_network import MLPClassifier


class MonteCarloNeuralNetwork(MonteCarloGameTree):
    """A player that estimates the value of states by using a Neural network.

    Instance Attributes:
        - root: Holds the GameState in the root of self.
        - value: Holds the value of the root state. This is None if the value has not been calculated yet.
        - children: Holds all subtrees of self connected to the root.
        - repeat: Holds the number of times a Monte Carlo tree search is performed to estimate the value of root.
    """

    root: GameState
    value: Optional[float]
    children: list[MonteCarloNeuralNetwork]
    repeat: int

    neural_network: MLPClassifier

    def __init__(self, start_state: GameState, neural_network: MLPClassifier,
                 repeat: int = 5, value: Optional[float] = None) -> None:
        super().__init__(start_state, repeat, value)
        self.neural_network = neural_network

    def expand_tree(self, state: GameState) -> None:
        """Add all children of state in self, if they are not already there.
        Adds a MinimaxGameTree instead of the generic GameTree

        Assumes that if some child is present, then all possible children are present.
        """
        if state == self.root:
            if self.children == []:
                self.children = [MonteCarloNeuralNetwork(move) for move in self.root.legal_moves()]
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

                return

        raise MoveNotLegalError(str(state.previous_move))

    def select_child(self) -> MonteCarloGameTree:
        """Chooses which state to explore in the exploration phase"""
        pass

    def move_value(self) -> float:
        """Estimate the value of the root by simulating possible games"""
        pass

    def update_value(self) -> None:
        """Update the value of self by using the values of the children in the backpropagation phase"""
        pass
