"""Holds the Monte Carlo Search Tree and Neural Network"""
from __future__ import annotations

import math
from typing import Optional

from Game import GameState, GameTree, MoveNotLegalError


class MonteCarloGameTree(GameTree):
    """A player that makes decisions using a monte carlo tree search.
    Instance Attributes:
        - root: Holds the GameState in the root of self.
        - value: Holds the value of the root state. This is None if the value has not been calculated yet.
        - children: Holds all subtrees of self connected to the root.
        - repeat: Holds the number of times a Monte Carlo tree search is performed to estimate the value of root.
    """
    root: GameState
    value: Optional[float]
    repeat: int
    children: list[MonteCarloSimulationGameTree]

    def __init__(self, start_state: GameState, repeat: int = 50, value: Optional[float] = None) -> None:
        super().__init__(start_state)
        self.value = value
        self.repeat = repeat

    def find_value(self) -> None:
        """Run a Monte Carlo tree search repeatedly to estimate the value the root."""
        for _ in range(self.repeat):
            self.monte_carlo_tree_search()

    def monte_carlo_tree_search(self) -> None:
        """Run a Monte Carlo tree search to update the value the root."""
        # Checks if self is a leaf
        if self.children != []:
            # Exploration phase
            explore_state = self.select_child()
            explore_state.find_value()
        else:
            # Expansion phase
            self.expand_root()

            # Simulation phase
            for child in self.children:
                child.value = child.move_value()

        # backpropagation phase
        if self.children == []:
            self.value = self.root.evaluate_position()
        else:
            self.update_value()

    def select_child(self) -> MonteCarloGameTree:
        """Chooses which state to explore in the exploration phase.

        Preconditions:
            - self.children != []
        """
        raise NotImplementedError

    def move_value(self) -> float:
        """Estimate the value of the root by simulating possible games in the simulation phase"""
        raise NotImplementedError

    def update_value(self) -> None:
        """Update the value of self by using the values of the children in the backpropagation phase"""
        raise NotImplementedError


class MonteCarloSimulationGameTree(MonteCarloGameTree):
    """A player that estimates the value of states by simulating possible games.

    Instance Attributes:
        - root: Holds the GameState in the root of self.
        - value: Holds the value of the root state. This is None if the value has not been calculated yet.
        - children: Holds all subtrees of self connected to the root.
        - repeat: Holds the number of times a Monte Carlo tree search is performed to estimate the value of root.

        - visits: Holds the number of times self has been simulated
        - wins: Holds the number of times self wins a simulation
        - exploration_parameter: Holds a value representing the proportion of times the AI chooses to
            explore rather than exploit.
    """

    root: GameState
    value: Optional[float]
    children: list[MonteCarloSimulationGameTree]
    repeat: int

    visits: int
    wins: int
    exploration_parameter: float

    def __init__(self, start_state: GameState, repeat: int = 50,
                 value: Optional[float] = None, exploration_parameter: float = 1.4142) -> None:
        super().__init__(start_state, repeat, value)

        self.visits = 0
        self.wins = 0
        self.exploration_parameter = exploration_parameter

    def expand_tree(self, state: GameState) -> None:
        """Add all children of state in self, if they are not already there.
        Adds a MinimaxGameTree instead of the generic GameTree

        Assumes that if some child is present, then all possible children are present.
        """
        if state == self.root:
            if self.children == []:
                self.children = [MonteCarloSimulationGameTree(move) for move in self.root.legal_moves()]
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
        """Chooses which state to explore in the exploration phase.

        Uses the upper confidence bound formula to choose which child should be searched.

        Preconditions:
            - self.children != []
        """

        explore = self.children[1]
        for child in self.children:
            if child.ucb(self.visits) > explore.ucb(self.visits):
                explore = child

        return explore

    def ucb(self, visits_parent: int) -> float:
        """A helper function returning the value used to check if a state is worth exploring,
        given the number of times the parent was visited.

        Uses the Upper Confidence Bound formula, as described here: """
        exploitation_value = self.wins / self.visits
        exploration_value = self.exploration_parameter * math.sqrt(math.log(visits_parent) / self.visits)

        return exploration_value + exploitation_value

    def move_value(self) -> float:
        """Estimate the value of the root by simulating possible games"""

    def update_value(self) -> None:
        """Update the value of self by using the values of the children in the backpropagation phase"""
        self.wins = 0
        self.visits = 0
        for child in self.children:
            self.wins += child.wins
            self.visits += child.visits

        self.value = self.wins / self.visits


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

    def __init__(self, start_state: GameState, repeat: int = 50, value: Optional[float] = None) -> None:
        super().__init__(start_state, repeat, value)

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
