"""Holds the Monte Carlo Search Tree

This file is Copyright (c) 2020 Mark Bedaywi
"""
from __future__ import annotations

import math
import random
from typing import Optional

from Player import RandomPlayer
from Game import GameState, GameTree, MoveNotLegalError, Player


class MonteCarloGameTree(GameTree):
    """A player that makes decisions using a monte carlo tree search.
    Instance Attributes:
        - root: Holds the GameState in the root of self.
        - value: Holds the value of the root state. This is 0 if the value has not been calculated yet.
        - children: Holds all subtrees of self connected to the root.
        - repeat: Holds the number of times a Monte Carlo tree search is performed to estimate the value of root.
        - exploration_parameter: Holds a value representing how much the AI should explore rather than exploit
        - visits: Holds the number of times self has been simulated
    """
    root: GameState
    children: list[MonteCarloSimulationGameTree]
    value: Optional[float]
    visits: int
    repeat: int
    exploration_parameter: float

    def __init__(self, start_state: GameState, repeat: int = 10000,
                 exploration_parameter: float = 1.4142, value: float = 0) -> None:
        super().__init__(start_state)
        self.value = value
        self.repeat = repeat
        self.exploration_parameter = exploration_parameter
        self.visits = 0

    def find_value(self) -> None:
        """Run a Monte Carlo tree search repeatedly to estimate the value the root."""
        for _ in range(self.repeat):
            self.monte_carlo_tree_search()

    def monte_carlo_tree_search(self) -> float:
        """Run a Monte Carlo tree search to update the value the root.

        Return the value added to backpropagate up the tree.
        """
        # Checks if self is a leaf
        if self.children != []:
            # Exploration phase
            explore_state = self.select_child()

            # We change the reward from 1 to 0 or 0 to 1, as the player changes
            reward = 1 - explore_state.monte_carlo_tree_search()
        else:
            # Expansion phase
            self.expand_root()

            # Simulation phase
            if self.children != []:
                child = random.choice(self.children)
                reward = child.move_value()
            else:
                reward = self.move_value()

        # backpropagation phase
        self.value += reward
        self.visits += 1

        return reward

    def select_child(self) -> MonteCarloGameTree:
        """Chooses which state to explore in the exploration phase.

        Preconditions:
            - self.children != []
        """
        explore = self.children[0]
        for child in self.children:
            if child.ucb(self.visits) > explore.ucb(self.visits):
                explore = child

        return explore

    def ucb(self, visits_parent: int) -> float:
        """Use the upper confidence bound to give a value
        representing to what extent a state is worth exploring."""
        raise NotImplementedError

    def move_value(self) -> float:
        """Estimate the value of the root by simulating possible games in the simulation phase"""
        raise NotImplementedError

    def copy(self) -> MonteCarloGameTree:
        """Return a copy of self"""
        raise NotImplementedError


class MonteCarloSimulationGameTree(MonteCarloGameTree):
    """A player that estimates the value of states by simulating possible games.

    Instance Attributes:
        - root: Holds the GameState in the root of self.
        - value: Holds the value of the root state. This is None if the value has not been calculated yet.
        - children: Holds all subtrees of self connected to the root.
        - repeat: Holds the number of times a Monte Carlo tree search is performed to estimate the value of root.
        - visits: Holds the number of times self has been simulated
        - exploration_parameter: Holds a value representing the proportion of times the AI chooses to
            explore rather than exploit.
    """

    root: GameState
    value: Optional[float]
    children: list[MonteCarloSimulationGameTree]
    repeat: int
    exploration_parameter: float
    visits: int

    def expand_tree(self, state: GameState) -> None:
        """Add all children of state in self, if they are not already there.
        Adds a MinimaxGameTree instead of the generic GameTree

        Assumes that if some child is present, then all possible children are present.
        """
        if state == self.root:
            if self.children == []:
                self.children = [
                    MonteCarloSimulationGameTree(
                        move,
                        self.repeat,
                        self.exploration_parameter
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
        """A helper function returning the value used to check if a state is worth exploring,
        given the number of times the parent was visited.

        Uses the Upper Confidence Bound formula, as described here: """
        if self.visits == 0:
            return float("inf")

        exploitation_value = self.value / self.visits
        exploration_value = self.exploration_parameter * math.sqrt(math.log(visits_parent) / self.visits)

        return exploration_value + exploitation_value

    def move_value(self) -> float:
        """"Play a game where players make random moves from self.
        Return 1 if the winner is player 1 and 0 if it is player 2, and 0.5 if it is a tie
        """
        random_player1 = RandomPlayer(self.root.copy())
        random_player2 = RandomPlayer(self.root.copy())
        game = self.root.game_type(random_player1, random_player2, self.root.copy())

        winner = game.play_game()[0]
        if winner[0]:  # If there was not a tie
            if winner[1] != self.root.turn:
                return 1
            else:
                return 0
        return 0.5

    def copy(self) -> MonteCarloSimulationGameTree:
        """Return a copy of self"""
        return MonteCarloSimulationGameTree(
            self.root.copy(),
            self.repeat,
            self.exploration_parameter,
            self.value
        )


class MonteCarloSimulationPlayer(Player):
    """A player that chooses the optimal move using a Monte Carlo search tree with simulation

    Instance Attributes:
        - game_tree: Holds the GameTree object the player uses to make decisions
        - repeat: Holds the number of times the MCTS is repeated before a decision is made
        - exploration_parameter: Holds the proportion of times the AI chooses to explore,
            as opposed to exploiting
    """
    game_tree: MonteCarloSimulationGameTree

    def __init__(self, start_state: GameState, game_tree: MonteCarloSimulationGameTree = None) -> None:
        if game_tree is not None:
            self.game_tree = game_tree
        else:
            self.game_tree = MonteCarloSimulationGameTree(start_state)

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

            # If it is player 1's turn, maximise
            if average_value > best_average_value:
                best_move = move
                best_average_value = best_move.value / best_move.visits

        return best_move.root

    def copy(self) -> MonteCarloSimulationPlayer:
        """Return a copy of self"""
        return MonteCarloSimulationPlayer(self.game_tree.root, self.game_tree.copy())
