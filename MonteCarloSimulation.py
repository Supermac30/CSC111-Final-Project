"""Holds the Monte Carlo Search Tree"""
from __future__ import annotations

import math
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
    value: Optional[float]
    repeat: int
    children: list[MonteCarloSimulationGameTree]
    exploration_parameter: float
    visits: int

    def __init__(self, start_state: GameState, repeat: int = 7,
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

    def update_value(self) -> None:
        """Update the value of self by using the values of the children in the backpropagation phase"""
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

        - is_player1: Is True if the player using the game tree is player 1, and False otherwise.
        - wins: Holds the number of times self wins a simulation
    """

    root: GameState
    value: Optional[float]
    children: list[MonteCarloSimulationGameTree]
    repeat: int
    exploration_parameter: float
    visits: int

    is_player1: bool
    wins: int

    def __init__(self, start_state: GameState, is_player1: bool, repeat: int = 5,
                 exploration_parameter: float = 1.4142, value: Optional[float] = None) -> None:
        super().__init__(start_state, repeat, exploration_parameter, value)

        self.num_of_simulations = 5
        self.is_player1 = is_player1
        self.wins = 0

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
                        self.is_player1,
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
                self.wins = child.wins
                self.visits = child.visits

                return

        raise MoveNotLegalError(str(state.previous_move))

    def ucb(self, visits_parent: int) -> float:
        """A helper function returning the value used to check if a state is worth exploring,
        given the number of times the parent was visited.

        Uses the Upper Confidence Bound formula, as described here: """
        exploitation_value = self.wins / self.visits
        exploration_value = self.exploration_parameter * math.sqrt(math.log(visits_parent) / self.visits)

        return exploration_value + exploitation_value

    def move_value(self) -> float:
        """Estimate the value of the root by simulating possible games.
        Updates self.wins and self.visits.
        """
        for _ in range(self.num_of_simulations):
            if self.simulate_game():
                self.wins += 1

        self.visits = self.num_of_simulations

        return self.wins / self.visits

    def simulate_game(self) -> bool:
        """Play a game where players make random moves from self
        and return whether the player in self won.
        """
        # There is no need to simulate if the game is over
        winner = self.root.winner()
        if winner is not None:
            if not winner[0]:
                return False
            return winner[1] == self.is_player1

        game_tree1 = GameTree(self.root.copy())
        game_tree2 = GameTree(self.root.copy())
        random_player1 = RandomPlayer(game_tree1)
        random_player2 = RandomPlayer(game_tree2)
        game = self.root.game_type(random_player1, random_player2, self.root.copy())

        winner = game.play_game()[0]
        if winner[0]:  # If there was not a tie
            return (winner[1] and self.is_player1) or (not winner[1] and not self.is_player1)
        return False

    def update_value(self) -> None:
        """Update the value of self by using the values of the children in the backpropagation phase"""
        self.wins = 0
        self.visits = self.num_of_simulations
        for child in self.children:
            self.wins += child.wins
            self.visits += child.visits

        self.value = self.wins / self.visits

    def copy(self) -> MonteCarloSimulationGameTree:
        """Return a copy of self"""
        return MonteCarloSimulationGameTree(
            self.root.copy(),
            self.is_player1,
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

    def __init__(self, game_tree: MonteCarloSimulationGameTree) -> None:
        super().__init__(game_tree)

    def choose_move(self) -> GameState:
        """Return the optimal move from the game state in self.game_tree.root

        Assumes the game is not over, that is, assumes there are possible
        legal moves from this position
        """
        best_move = self.game_tree.children[0]
        for move in self.game_tree.children:
            move.find_value()

            if move.value > best_move.value:
                best_move = move

        return best_move.root

    def copy(self) -> MonteCarloSimulationPlayer:
        """Return a copy of self"""
        return MonteCarloSimulationPlayer(self.game_tree.copy())
