""" Holds the main function

This file is Copyright (c) 2020 Mark Bedaywi
"""
from Menu import Menu
import tkinter as tk

if __name__ == '__main__':
    screen = tk.Tk()
    menu = Menu(screen)
    menu.mainloop()
