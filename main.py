""" Holds the main function

This file is Copyright (c) 2020 Mark Bedaywi
"""
from menu import Menu
import tkinter as tk

if __name__ == '__main__':
    screen = tk.Tk()
    screen.title("Traversing Game Trees Intelligently")
    screen.minsize(400, 130)
    menu = Menu(screen)
    menu.mainloop()
