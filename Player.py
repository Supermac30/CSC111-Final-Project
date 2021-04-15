""" Stores the Minimax Player.

This file is Copyright (c) 2020 Mark Bedaywi
"""
from __future__ import annotations
import random
from typing import Optional, Tuple, Dict
from Game import GameState, Player, GameTree, MoveNotLegalError


class RandomPlayer(Player):
    """A player that makes random moves for the purpose of testing"""
    def __init__(self, start_state: GameState, game_tree: GameTree = None):
        if game_tree is not None:
            self.game_tree = game_tree
        else:
            self.game_tree = GameTree(start_state)

    def choose_move(self) -> GameState:
        """Return a random move from the game state"""
        possible_moves = [child.root for child in self.game_tree.children]

        return random.choice(possible_moves)

    def copy(self) -> RandomPlayer:
        """Return a copy of self"""
        return RandomPlayer(self.game_tree.root, self.game_tree.copy())


class MinimaxGameTree(GameTree):
    """A GameTree that stores the value of the root for the purpose of minimax

     Instance Attributes:
        - root: Holds the GameState in the root of self.
        - value: Holds the value of the root state. This is None if the value has not been calculated yet.
        - children: Holds all subtrees of self connected to the root.
    """
    root: GameState
    value: Optional[float]
    children: list[MinimaxGameTree]

    def __init__(self, start_state: GameState, value: Optional[float] = None) -> None:
        super().__init__(start_state)
        self.value = value

    def find_value(self, memoize: Dict[Tuple[int, str, float, float], float], depth: int = -1,
                   alpha: float = -float('inf'), beta: float = float('inf')) -> None:
        """Runs the minimax algorithm to update the value the root.

        memoize stores the value of each state to avoid re-computation. It is a map
        from a tuple storing the depth searched to and the string representation
        of the state into the value calculated

        Uses alpha beta pruning to remove moves that are too bad (or too good for player 1, if it is player 2s turn)
        to bother searching through, relative to moves already searched through.

        If depth is not negative, then minimax is only run up to the specified depth."""

        # Note: We have to store the alpha and beta values when memoizing. This stumped me
        # for a couple of hours as the AI would occasionally not make the optimal move,
        # until I realized that the alpha and beta values depend on the surrounding moves,
        # and since they can be different, it doesn't make sense to treat
        # the stored values in the same way.

        # Storing the depth doesn't matter if a full search is done
        if depth < 0:
            state_repr = (-1, str(self.root), alpha, beta)
        # Stores the depth so that less accurate results are never used
        else:
            state_repr = (depth, str(self.root), alpha, beta)

        # Doesn't recompute the value of the state if it has been seen before
        if state_repr in memoize:
            self.value = memoize[state_repr]
            return

        if depth == 0 or self.root.winner() is not None:
            self.value = self.root.evaluate_position()
            return

        self.expand_root()
        # Maximizes the value
        if self.root.turn:
            # Finds the value of each child
            for child in self.children:
                child.find_value(memoize, depth - 1, alpha, beta)

                alpha = max(alpha, child.value)

                # If a better move has been seen before
                if alpha >= beta:
                    self.value = beta

                    # Memoizes the value of the state
                    memoize[state_repr] = self.value
                    return

            self.value = alpha

        # Minimizes the value
        else:
            # Finds the value of each child
            for child in self.children:
                child.find_value(memoize, depth - 1, alpha, beta)

                beta = min(beta, child.value)

                # If a worse move has been seen before
                if alpha >= beta:
                    self.value = alpha

                    # Memoizes the value of the state
                    memoize[state_repr] = self.value
                    return

            self.value = beta

        # Memoizes the value of the state
        memoize[state_repr] = self.value

    def expand_tree(self, state: GameState) -> None:
        """Add all children of state in self, if they are not already there.
        Adds a MinimaxGameTree instead of the generic GameTree

        Assumes that if some child is present, then all possible children are present.
        """
        if state == self.root:
            if self.children == []:
                self.children = [MinimaxGameTree(move) for move in self.root.legal_moves()]
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

    def copy(self) -> MinimaxGameTree:
        """Return a copy of self"""
        return MinimaxGameTree(self.root.copy(), self.value)


class MinimaxPlayer(Player):
    """A player that chooses the optimal move using the minimax algorithm

    Instance Attributes:
        - game_tree: Holds the GameTree object the player uses to make decisions
        - depth: Holds the depth that the search will be made to
        - memoize: Holds the memoized values for states
    """
    game_tree: MinimaxGameTree
    depth: int
    memoize: dict[Tuple[int, str, float, float], float]

    def __init__(self, start_state: GameState, game_tree: MinimaxGameTree = None, depth: int = -1) -> None:
        self.depth = depth
        self.memoize = {}
        if game_tree is not None:
            self.game_tree = game_tree
        else:
            self.game_tree = MinimaxGameTree(start_state)

    def choose_move(self) -> GameState:
        """Return the optimal move from the game state in self.game_tree.root

        Assumes the game is not over, that is, assumes there are possible
        legal moves from this position
        """
        turn = self.game_tree.root.turn

        best_move = self.game_tree.children[0]
        for move in self.game_tree.children:
            move.find_value(self.memoize, self.depth)

            # If it is player 1's turn, maximise
            if turn and move.value > best_move.value:
                best_move = move
            # If it is player 2's turn, minimise
            elif not turn and move.value < best_move.value:
                best_move = move

        return best_move.root

    def copy(self) -> MinimaxPlayer:
        """Return a copy of self"""
        return MinimaxPlayer(self.game_tree.root, self.game_tree.copy(), self.depth)
