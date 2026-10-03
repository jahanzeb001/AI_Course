
import tkinter as tk
from tkinter import messagebox


# ==========================================
# TIC-TAC-TOE GAME
# ==========================================

# Current player
current_player = "X"

# Store the game board
board = [""] * 9


# ==========================================
# CHECK WINNER
# ==========================================

def check_winner():

    # All possible winning combinations
    winning_combinations = [
        (0, 1, 2),  # Top row
        (3, 4, 5),  # Middle row
        (6, 7, 8),  # Bottom row
        (0, 3, 6),  # Left column
        (1, 4, 7),  # Middle column
        (2, 5, 8),  # Right column
        (0, 4, 8),  # Diagonal
        (2, 4, 6)   # Diagonal
    ]

    # Check every winning combination
    for a, b, c in winning_combinations:

        if board[a] != "" and board[a] == board[b] == board[c]:
            return board[a]

    # Check for draw
    if "" not in board:
        return "Draw"

    # Game is still running
    return None


# ==========================================
# BUTTON CLICK
# ==========================================

def button_click(index):

    global current_player

    # IMPORTANT:
    # Don't allow player to change an already
    # selected box
    if board[index] != "":
        return

    # Put X or O on the board
    board[index] = current_player

    # Display X or O on the button
    buttons[index].config(text=current_player)

    # Check if somebody won
    result = check_winner()

    if result == "X":
        messagebox.showinfo("Game Over", "Player X Wins! 🎉")
        reset_game()
        return

    elif result == "O":
        messagebox.showinfo("Game Over", "Player O Wins! 🎉")
        reset_game()
        return

    elif result == "Draw":
        messagebox.showinfo("Game Over", "It's a Draw! 🤝")
        reset_game()
        return

    # Switch player
    if current_player == "X":
        current_player = "O"
    else:
        current_player = "X"

    # Update player label
    player_label.config(
        text=f"Player {current_player}'s Turn"
    )


# ==========================================
# RESET GAME
# ==========================================

def reset_game():

    global current_player

    # Reset board
    board.clear()
    board.extend([""] * 9)

    # Start again with X
    current_player = "X"

    # Clear all buttons
    for button in buttons:
        button.config(text="")

    # Update label
    player_label.config(
        text="Player X's Turn"
    )


# ==========================================
# MAIN WINDOW
# ==========================================

root = tk.Tk()

root.title("Tic-Tac-Toe")

root.geometry("450x550")

root.resizable(False, False)


# ==========================================
# TITLE
# ==========================================

title_label = tk.Label(
    root,
    text="⭕ TIC-TAC-TOE ❌",
    font=("Arial", 28, "bold")
)

title_label.pack(pady=20)


# ==========================================
# PLAYER LABEL
# ==========================================

player_label = tk.Label(
    root,
    text="Player X's Turn",
    font=("Arial", 18, "bold")
)

player_label.pack(pady=10)


# ==========================================
# GAME BOARD
# ==========================================

game_frame = tk.Frame(root)

game_frame.pack(pady=20)


# Store buttons
buttons = []


# Create 9 buttons
for i in range(9):

    button = tk.Button(
        game_frame,
        text="",
        font=("Arial", 32, "bold"),
        width=5,
        height=2,
        command=lambda index=i: button_click(index)
    )

    # Arrange buttons in 3x3 grid
    button.grid(
        row=i // 3,
        column=i % 3,
        padx=5,
        pady=5
    )

    buttons.append(button)


# ==========================================
# RESET BUTTON
# ==========================================

reset_button = tk.Button(
    root,
    text="New Game",
    font=("Arial", 14, "bold"),
    width=15,
    command=reset_game
)

reset_button.pack(pady=20)


# ==========================================
# EXIT BUTTON
# ==========================================

exit_button = tk.Button(
    root,
    text="Exit",
    font=("Arial", 12),
    width=10,
    command=root.destroy
)

exit_button.pack()


# ==========================================
# START GAME
# ==========================================

root.mainloop()

