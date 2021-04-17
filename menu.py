""" Stores the Menu for playing the games.

This file is Copyright (c) 2020 Mark Bedaywi
"""
from typing import Type
import tkinter as tk

from sklearn.neural_network import MLPClassifier

import game

import tic_tac_toe
import reversi
import connect_four

import minimax_player
import monte_carlo_simulation
import monte_carlo_neural_network
import openings_player


class Menu(tk.Frame):
    """Stores the Menu to launch desired games

    Instance Attributes:
        - game_state: The type of game state that will be used
        - player1_id: Stores an integer representing the type of player player 1 will be
        - player2_id: Stores an integer representing the type of player player 2 will be

        - title: The title of the current menu
        - game_buttons: Holds the Tkinter buttons displaying games
        - player_buttons: Holds the Tkinter buttons displaying players

        - accept_opening: The buttons used to make a player use an opening
        - depth_entry: The text box for entering the desired depth of the minimax player
        - depth_button: The button for inputting the desired depth of the minimax player
        - repetition_entry: The text box for entering the
            desired amount of repetitions of the MCST player
        - repetition_button: The button for inputting the
            desired amount of repetitions of the MCST player

        - with_opening: The value at index i is true if the ith player should use an opening book
        - depth: The value at the index i is the depth of the
            ith player if they use a minimax search tree
        - repetitions: The value at index i is the number of repetitions of
            the ith player if they use a MCTS.

        - choose_player1: Whether player 1 has been chosen yet
    """
    game_state: Type[game.GameState]
    player1_id: int
    player2_id: int

    title: tk.Label
    game_buttons: list[tk.Button]
    player_buttons: list[tk.Button]

    opening_buttons: list[tk.Button]
    depth_entry: tk.Entry
    depth_button: tk.Button
    repetition_entry: tk.Entry
    repetition_button: tk.Button

    with_opening: list[bool, bool]
    depth: list[int, int]
    repetitions: list[int, int]

    choose_player1: bool

    def __init__(self, master: tk.Tk = None) -> None:
        self.game_buttons = []
        self.player_buttons = []
        self.opening_buttons = []
        self.choose_player1 = True
        self.with_opening = [False, False]
        self.depth = [0, 0]
        self.repetitions = [0, 0]
        self.player1_id = 0
        self.player2_id = 0

        super().__init__(master)
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

            # Inputting i directly doesn't result in the required behaviour as the value of i after
            # the end of the loop is used.
            if i == 0:
                self.game_buttons[i]['command'] = lambda: self.assign_game(0)
            elif i == 1:
                self.game_buttons[i]['command'] = lambda: self.assign_game(1)
            else:
                self.game_buttons[i]['command'] = lambda: self.assign_game(2)
            self.game_buttons[i].pack(side='top')

    def assign_game(self, game_id: int) -> None:
        """Assigns the game to be played and changes the menu"""
        if game_id == 0:
            self.game_state = tic_tac_toe.TicTacToeGameState
        elif game_id == 1:
            self.game_state = connect_four.ConnectFourGameState
        elif game_id == 2:
            self.game_state = reversi.ReversiGameState

        self.create_player_menu()

    def create_player_menu(self) -> None:
        """Displays the menu where a player is chosen"""
        title = 'Choose Player '
        if self.choose_player1:
            title += '1'
        else:
            title += '2'
        self.title['text'] = title
        for game_button in self.game_buttons:
            game_button.destroy()

        self.pack()

        players = ['Random Player', 'Minimax Player', 'MCTS Simulation Player',
                   'MCTS Neural Network Player', 'Neural Network Player']

        # There can only be one human player
        if self.player1_id != 5:
            players.append('Human Player')

        for i in range(len(players)):
            self.player_buttons.append(tk.Button(self))
            self.player_buttons[i]['text'] = players[i]
            if i == 0:
                self.player_buttons[i]['command'] = lambda: self.assign_player(0)
            elif i == 1:
                self.player_buttons[i]['command'] = lambda: self.assign_player(1)
            elif i == 2:
                self.player_buttons[i]['command'] = lambda: self.assign_player(2)
            elif i == 3:
                self.player_buttons[i]['command'] = lambda: self.assign_player(3)
            elif i == 4:
                self.player_buttons[i]['command'] = lambda: self.assign_player(4)
            else:
                self.player_buttons[i]['command'] = lambda: self.assign_player(5)
            self.player_buttons[i].pack(side='top')

    def assign_player(self, player_id: int) -> None:
        """Assigns the player."""
        for player_button in self.player_buttons:
            player_button.destroy()
        self.player_buttons = []

        if self.choose_player1:
            self.choose_player1 = False
            self.player1_id = player_id
            if self.game_state == reversi.ReversiGameState and player_id != 5:
                self.choose_opening(True)
            elif player_id == 1:
                self.choose_depth(True)
            elif player_id in (2, 3):
                self.choose_repetition(True)
            else:
                self.create_player_menu()
        else:
            self.player2_id = player_id
            if self.game_state == reversi.ReversiGameState and player_id != 5:
                self.choose_opening(False)
            elif player_id == 1:
                self.choose_depth(False)
            elif player_id in (2, 3):
                self.choose_repetition(False)
            else:
                self.start_game()

    def choose_opening(self, is_player_1: bool) -> None:
        """Displays the menu where the user can choose whether
        the chosen player should use an opening book"""
        self.opening_buttons.append(tk.Button(self))
        self.opening_buttons[0]['text'] = "Make Player use openings book"
        self.opening_buttons[0]['command'] = lambda: self.assign_opening(True, is_player_1)
        self.opening_buttons[0].pack(side='top')

        self.opening_buttons.append(tk.Button(self))
        self.opening_buttons[1]['text'] = "Don't Make Player use openings book"
        self.opening_buttons[1]['command'] = lambda: self.assign_opening(False, is_player_1)
        self.opening_buttons[1].pack(side='top')

    def assign_opening(self, value: bool, is_player1: bool) -> None:
        """Assign the value of self.with_opening"""
        self.opening_buttons[0].destroy()
        self.opening_buttons[1].destroy()
        self.opening_buttons = []

        if is_player1:
            self.with_opening[0] = value
            if self.player1_id == 1:
                self.choose_depth(is_player1)
            elif self.player1_id == 2 or self.player1_id == 3:
                self.choose_repetition(is_player1)
            else:
                self.create_player_menu()
        else:
            self.with_opening[1] = value
            if self.player2_id == 1:
                self.choose_depth(is_player1)
            elif self.player2_id == 2 or self.player2_id == 3:
                self.choose_repetition(is_player1)
            else:
                self.start_game()

    def choose_depth(self, is_player1: bool) -> None:
        """Displays the menu where the user can choose the depth of the chosen minimax player"""
        self.title['text'] = "Choose the depth of the Minimax Player " \
                             "(input -1 if the maximum depth is infinite)"

        self.depth_entry = tk.Entry(self)
        self.depth_entry.pack(side='top')

        self.depth_button = tk.Button(self)
        self.depth_button['text'] = "Input"
        self.depth_button['command'] = lambda: self.assign_depth(is_player1)
        self.depth_button.pack(side='top')

    def assign_depth(self, is_player1: bool) -> None:
        """Assign the value of depth to the right index depending on
        the value of is_player1"""
        value = int(self.depth_entry.get())
        self.depth_entry.destroy()
        self.depth_button.destroy()

        if is_player1:
            self.depth[0] = value
            self.create_player_menu()
        else:
            self.depth[1] = value
            self.start_game()

    def choose_repetition(self, is_player1: bool) -> None:
        """Displays the menu where the user can choose the repetition of the chosen MCST player"""
        self.title['text'] = "Choose the number of tree searches per turn of the MCST Player"

        self.repetition_entry = tk.Entry(self)
        self.repetition_entry.pack(side='top')

        self.repetition_button = tk.Button(self)
        self.repetition_button['text'] = "Input"
        self.repetition_button['command'] = lambda: self.assign_repetition(is_player1)
        self.repetition_button.pack(side='top')

    def assign_repetition(self, is_player1: bool) -> None:
        """Assign the value of depth to the right index depending on
                the value of is_player1"""
        value = int(self.repetition_entry.get())
        self.repetition_entry.destroy()
        self.repetition_button.destroy()

        if is_player1:
            self.repetitions[0] = value
            self.create_player_menu()
        else:
            self.repetitions[1] = value
            self.start_game()

    def get_player(self, player_id: int, neural_network: MLPClassifier,
                   is_player_1: bool) -> minimax_player:
        """Use the inputs to return the player chosen"""
        start_state = self.game_state()

        if is_player_1:
            index = 0
        else:
            index = 1

        if player_id == 0:
            player = minimax_player.RandomPlayer(start_state.copy())
        elif player_id == 1:
            player = minimax_player.MinimaxPlayer(start_state.copy(), depth=self.depth[index])
        elif player_id == 2:
            player = monte_carlo_simulation.MonteCarloSimulationPlayer(
                start_state.copy(), repeat=self.repetitions[index])
        elif player_id == 3:
            player = monte_carlo_neural_network.MonteCarloNeuralNetworkPlayer(
                start_state.copy(),
                neural_network,
                repeat=self.repetitions[index]
            )
        else:
            player = monte_carlo_neural_network.NeuralNetworkPlayer(
                start_state.copy(),
                neural_network,
                is_player_1
            )

        if self.with_opening[0]:
            player = openings_player.ReversiOpeningsPlayer(start_state.copy(), player)

        return player

    def start_game(self) -> None:
        """Starts the game"""
        if self.game_state == tic_tac_toe.TicTacToeGameState:
            file_name = "data/neural_networks/TicTacToeNeuralNetwork.txt"
        elif self.game_state == connect_four.ConnectFourGameState:
            file_name = "data/neural_networks/ConnectFourNeuralNetwork.txt"
        else:  # self.game_state == Reversi.ReversiGameState
            file_name = "data/neural_networks/ReversiNeuralNetwork.txt"
        neural_network = monte_carlo_neural_network.load_neural_network(file_name)

        player1 = self.get_player(self.player1_id, neural_network, True)
        player2 = self.get_player(self.player2_id, neural_network, False)

        created_game = game.Game(player1, player2)
        statistics_game = created_game.copy()

        # Play the displayed game
        if self.player1_id == 5:
            history = created_game.play_with_human(True)[1]
        elif self.player2_id == 5:
            history = created_game.play_with_human(False)[1]
        else:
            history = created_game.play_game(False)[1]

        # Display the game
        game.display_game(history)

        # If a human player is not chosen
        if self.player1_id != 5 and self.player2_id != 5:
            # Calculate Statistics
            player_1_wins, player_2_wins = statistics_game.play_games(5)

            # Display Statistics
            self.title['text'] = 'In a simulation of 5 games: \nPlayer 1 won ' + \
                                 str(player_1_wins) + ' times.\nPlayer 2 won ' + \
                                 str(player_2_wins) + ' times.'


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        'max-line-length': 100,
        'disable': ['E1136']
    })
