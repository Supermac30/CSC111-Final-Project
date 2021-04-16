""" Stores the Menu for playing the games.

This file is Copyright (c) 2020 Mark Bedaywi
"""
from typing import Type, Tuple
import tkinter as tk

import Game

import TicTacToe
import Reversi
import ConnectFour

import Player
import MonteCarloSimulation
import MonteCarloNeuralNetwork
import OpeningsPlayer


class Menu(tk.Frame):
    """Stores the Menu to launch desired games

    Instance Attributes:
        - game_state: The type of game state that will be used
        - player1_id: Stores an integer representing the type of player player 1 will be
        - player2_id: Stores an integer representing the type of player player 2 will be

        - game_buttons: Holds the Tkinter buttons displaying games
        - player_buttons: Holds the Tkinter buttons displaying players

        - with_opening: The value at index i is true if the ith player should use an opening book
        - depth: The value at the index i is the depth of the ith player if they use a minimax search tree
        - repetitions: The value at index i is the number of repetitions of the ith player if they
            use a MCTS.

        - choose_player1: Whether player 1 has been chosen yet
    """
    game_state: Type[Game.GameState]
    player1_id: int
    player2_id: int

    title: tk.Label
    game_buttons: list[tk.Button]
    player_buttons: list[tk.Button]

    accept_opening: tk.Button
    decline_opening: tk.Button

    with_opening: list[bool, bool]
    depth: Tuple[int, int]
    repetitions: Tuple[int, int]

    choose_player1: bool

    def __init__(self, master=None) -> None:
        self.game_buttons = []
        self.player_buttons = []
        self.choose_player1 = True
        self.with_opening = [False, False]

        super().__init__(master)
        self.master = master
        self.pack()
        self.create_game_menu()

    def create_game_menu(self) -> None:
        """Displays the menu where a game is chosen"""
        self.title = tk.Label(self, text="Choose Game")
        self.title.pack(side='top')

        games = ['Tic Tac Toe', 'Connect Four', 'Reversi']
        for i in range(3):
            self.game_buttons.append(tk.Button(self))
            self.game_buttons[i]['text'] = games[i]
            self.game_buttons[i]['command'] = lambda: self.assign_game(i)
            self.game_buttons[i].pack(side='top')

    def create_player_menu(self) -> None:
        """Displays the menu where a player is chosen"""
        self.title['text'] = 'Choose Player 1'
        for game_button in self.game_buttons:
            game_button.destroy()

        self.pack()

        players = ['Random Player', 'Minimax Player', 'MCTS Simulation Player',
                   'MCTS Neural Network Player', 'Neural Network Player']
        for i in range(5):
            self.player_buttons.append(tk.Button(self))
            self.player_buttons[i]['text'] = players[i]
            self.player_buttons[i]['command'] = lambda: self.assign_player(i)
            self.player_buttons[i].pack(side='top')

    def assign_game(self, game_id: int) -> None:
        """Assigns the game to be played and changes the menu"""
        if game_id == 0:
            self.game_state = TicTacToe.TicTacToeGameState
        elif game_id == 1:
            self.game_state = ConnectFour.ConnectFourGameState
        elif game_id == 2:
            self.game_state = Reversi.ReversiGameState

        self.pack_forget()
        self.create_player_menu()

    def assign_player(self, player_id: int) -> None:
        """Assigns the player."""
        if self.choose_player1:
            self.choose_player1 = False
            self.player1_id = player_id
            self.pack_forget()
            if self.game_state == Reversi.ReversiGameState:
                self.choose_opening(True)
            elif player_id == 1:
                self.choose_depth()
            elif player_id == 2 or player_id == 3:
                self.choose_repetition()
            else:
                self.create_player_menu()
        else:
            self.player2_id = player_id
            self.pack_forget()
            if self.game_state == Reversi.ReversiGameState:
                self.choose_opening(True)
            elif player_id == 1:
                self.choose_depth()
            elif player_id == 2 or player_id == 3:
                self.choose_repetition()
            else:
                self.start_game()

    def choose_opening(self, is_player_1: bool) -> None:
        """Displays the menu where the user can choose whether the chosen player should use an opening book"""
        self.accept_opening = tk.Button(self)
        self.accept_opening['text'] = "Make Player use openings book"
        self.accept_opening['command'] = lambda: self.assign_opening(True, is_player_1)
        self.accept_opening.pack(side='top')

        self.decline_opening = tk.Button(self)
        self.decline_opening['text'] = "Don't Make Player use openings book"
        self.decline_opening['command'] = lambda: self.assign_opening(False, is_player_1)
        self.decline_opening.pack(side='top')

    def assign_opening(self, value: bool, is_player_1: bool):
        """Assign the value of self.with_opening"""
        self.pack_forget()
        if is_player_1:
            self.with_opening[0] = value
            if self.player1_id == 1:
                self.choose_depth()
            elif self.player1_id == 2 or self.player1_id == 3:
                self.choose_repetition()
            else:
                self.create_player_menu()
        else:
            self.with_opening[1] = value
            if self.player2_id == 1:
                self.choose_depth()
            elif self.player2_id == 2 or self.player2_id == 3:
                self.choose_repetition()
            else:
                self.start_game()

    def choose_depth(self) -> None:
        """Displays the menu where the user can choose the depth of the chosen minimax player"""
        # TODO

    def assign_depth(self) -> None:

    def choose_repetition(self) -> None:
        """Displays the menu where the user can choose the repetition of the chosen MCST player"""
        # TODO

    def assign_repetition(self) -> None:

    def start_game(self) -> None:
        """Starts the game"""
        import GameGUI
        from Game import Game

        if self.game_state == TicTacToe.TicTacToeGameState:
            file_name = "TicTacToeNeuralNetwork.txt"
        elif self.game_state == ConnectFour.ConnectFourGameState:
            file_name = "ConnectFourNeuralNetwork.txt"
        else:  # self.game_state == Reversi.ReversiGameState
            file_name = "ReversiNeuralNetwork.txt"
        neural_network = MonteCarloNeuralNetwork.load_neural_network(file_name)
        start_state = self.game_state()

        if self.player1_id == 0:
            player1 = Player.RandomPlayer(start_state.copy())
        elif self.player1_id == 1:
            player1 = Player.MinimaxPlayer(start_state.copy(), depth=self.depth[0])
        elif self.player1_id == 2:
            player1 = MonteCarloSimulation.MonteCarloSimulationPlayer(start_state.copy(), repeat=self.repetitions[0])
        elif self.player1_id == 3:
            player1 = MonteCarloNeuralNetwork.MonteCarloNeuralNetworkPlayer(
                start_state.copy(),
                neural_network,
                True,
                repeat=self.repetitions[0]
            )
        else:  # self.player1_id == 4
            player1 = MonteCarloNeuralNetwork.MonteCarloNeuralNetwork(
                start_state.copy(),
                neural_network,
                True
            )

        if self.with_opening[0]:
            player1 = OpeningsPlayer.ReversiOpeningsPlayer(start_state.copy(), player1)

        if self.player2_id == 0:
            player2 = Player.RandomPlayer(start_state.copy())
        elif self.player2_id == 1:
            player2 = Player.MinimaxPlayer(start_state.copy(), depth=self.depth[1])
        elif self.player2_id == 2:
            player2 = MonteCarloSimulation.MonteCarloSimulationPlayer(start_state.copy(), repeat=self.repetitions[1])
        elif self.player2_id == 3:
            player2 = MonteCarloNeuralNetwork.MonteCarloNeuralNetworkPlayer(
                start_state.copy(),
                neural_network,
                False,
                repeat=self.repetitions[1]
            )
        else:  # self.player2_id == 4
            player2 = MonteCarloNeuralNetwork.MonteCarloNeuralNetwork(
                start_state.copy(),
                neural_network,
                False
            )

        if self.with_opening[1]:
            player2 = OpeningsPlayer.ReversiOpeningsPlayer(start_state.copy(), player2)

        game = Game(player1, player2)
        winner, history = game.play_game()
        GameGUI.display_game(history)
