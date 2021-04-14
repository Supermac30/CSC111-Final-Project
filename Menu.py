""" Stores the Menu for playing the games.

This file is Copyright (c) 2020 Mark Bedaywi
"""
from typing import Type
import tkinter as tk

import Game

import TicTacToe
import Reversi
import ConnectFour

import Player
import MonteCarloSimulation
import MonteCarloNeuralNetwork


class Menu(tk.Frame):
    """Stores the Menu to launch desired games

    Instance Attributes:
        - game: The type of game that will be played
        - game_state: The type of game state that will be used
        - player1: The type of player player 1 will be
        - player2: The type of player player 2 will be

        - game_buttons: Holds the Tkinter buttons displaying games
        - player_buttons: Holds the Tkinter buttons displaying players

        - choose_player1: Whether player 1 has been chosen yet
    """
    game: Type[Game.Game]
    game_state: Type[Game.GameState]
    player1: Type[Game.Player]
    player2: Type[Game.Player]

    title: tk.Label
    game_buttons: list[tk.Button]
    player_buttons: list[tk.Button]

    choose_player1: bool

    def __init__(self, master=None) -> None:
        self.game_buttons = []
        self.player_buttons = []
        self.choose_player1 = True

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

        players = ['Random Player', 'Minimax Player', 'MCTS Simulation Player', 'MCTS Neural Network Player']
        for i in range(4):
            self.player_buttons.append(tk.Button(self))
            self.player_buttons[i]['text'] = players[i]
            self.player_buttons[i]['command'] = lambda: self.assign_player(i)
            self.player_buttons[i].pack(side='top')

    def assign_game(self, game_id: int) -> None:
        """Assigns the game to be played and changes the menu"""
        if game_id == 0:
            self.game = TicTacToe.TicTacToe
            self.game_state = TicTacToe.TicTacToeGameState
        elif game_id == 1:
            self.game = ConnectFour.ConnectFour
            self.game_state = ConnectFour.ConnectFourGameState
        elif game_id == 2:
            self.game = Reversi.Reversi
            self.game_state = Reversi.ReversiGameState

        self.pack_forget()
        self.create_player_menu()

    def assign_player(self, player_id: int) -> None:
        """Assigns the player."""
        if self.choose_player1:
            if player_id == 0:
                self.player1 = Player.RandomPlayer
            elif player_id == 1:
                self.player1 = Player.MinimaxPlayer
            elif player_id == 2:
                self.player1 = MonteCarloSimulation.MonteCarloSimulationPlayer
            elif player_id == 3:
                self.player1 = MonteCarloNeuralNetwork.MonteCarloNeuralNetworkPlayer

            self.choose_player1 = False
            self.title['text'] = 'Choose Player 2'
        else:
            if player_id == 0:
                self.player2 = Player.RandomPlayer
            elif player_id == 1:
                self.player2 = Player.MinimaxPlayer
            elif player_id == 2:
                self.player2 = MonteCarloSimulation.MonteCarloSimulationPlayer
            elif player_id == 3:
                self.player2 = MonteCarloNeuralNetwork.MonteCarloNeuralNetworkPlayer

            self.start_game()

    def start_game(self) -> None:
        """Starts the game"""
        # TODO
