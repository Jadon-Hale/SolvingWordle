import tkinter as tk
from tkinter import messagebox
import random

# A sample list of 5-letter words.
WORDS = [
    "apple", "brick", "crane", "drape", "eagle", "flute", "glide",
    "hoist", "irony", "joker", "knife", "lemon", "mango", "nerve",
    "ocean", "piano", "quart", "raven", "shard", "tiger", "unity",
    "vivid", "wrist", "xenon", "yacht", "zebra"
]

# Choose a random secret word from the list.
secret_word = random.choice(WORDS)
# Uncomment the next line to print the secret word for debugging.
# print("Secret word:", secret_word)

# Color definitions (feel free to adjust these to your liking).
COLOR_GREEN = "#6aaa64"   # correct letter & position
COLOR_YELLOW = "#c9b458"  # letter is in the word, wrong position
COLOR_GRAY = "#787c7e"    # letter not in word
COLOR_EMPTY = "white"     # background for empty cells

# Create the main window.
root = tk.Tk()
root.title("Wordle")

# Create a frame for the grid of letter cells.
grid_frame = tk.Frame(root)
grid_frame.pack(pady=20)

# Create a 6x5 grid (6 attempts, 5 letters each) using Label widgets.
rows, cols = 6, 5
label_grid = [[None for _ in range(cols)] for _ in range(rows)]
for r in range(rows):
    for c in range(cols):
        lbl = tk.Label(
            grid_frame, text="", width=4, height=2,
            font=("Helvetica", 24), borderwidth=2, relief="solid",
            bg=COLOR_EMPTY
        )
        lbl.grid(row=r, column=c, padx=5, pady=5)
        label_grid[r][c] = lbl

# Create an input frame for the guess entry and submit button.
input_frame = tk.Frame(root)
input_frame.pack(pady=10)

guess_entry = tk.Entry(input_frame, font=("Helvetica", 18))
guess_entry.grid(row=0, column=0, padx=5)
guess_entry.focus()  # set focus on the entry widget

submit_button = tk.Button(input_frame, text="Submit", font=("Helvetica", 14))
submit_button.grid(row=0, column=1, padx=5)

# Track the current row (attempt).
current_row = 0

def get_colors(guess, secret):
    """
    Compare the guess to the secret word and return a list of color codes.
    'green' for correct position, 'yellow' for letter in word but wrong
    position, and 'gray' for letters not in the word.
    """
    colors = [""] * 5
    # Count the letters in the secret word.
    secret_letter_counts = {}
    for letter in secret:
        secret_letter_counts[letter] = secret_letter_counts.get(letter, 0) + 1

    # First pass: Mark correct letters (green).
    for i in range(5):
        if guess[i] == secret[i]:
            colors[i] = "green"
            secret_letter_counts[guess[i]] -= 1

    # Second pass: Mark letters that are in the word but in the wrong position.
    for i in range(5):
        if colors[i] == "":
            if guess[i] in secret_letter_counts and secret_letter_counts[guess[i]] > 0:
                colors[i] = "yellow"
                secret_letter_counts[guess[i]] -= 1
            else:
                colors[i] = "gray"
    return colors

def submit_guess(event=None):
    global current_row
    if current_row >= rows:
        return  # No more guesses allowed.
        
    guess = guess_entry.get().lower()
    if len(guess) != 5 or not guess.isalpha():
        messagebox.showinfo("Invalid Guess", "Please enter a 5-letter word.")
        return

    # Clear the entry for the next guess.
    guess_entry.delete(0, tk.END)

    colors = get_colors(guess, secret_word)

    # Update the grid with the guessed letters and their colors.
    for i in range(5):
        letter = guess[i].upper()
        lbl = label_grid[current_row][i]
        lbl.config(text=letter)
        if colors[i] == "green":
            lbl.config(bg=COLOR_GREEN, fg="white")
        elif colors[i] == "yellow":
            lbl.config(bg=COLOR_YELLOW, fg="white")
        else:
            lbl.config(bg=COLOR_GRAY, fg="white")
    
    # Check if the guess is correct.
    if guess == secret_word:
        messagebox.showinfo("Congratulations!", "You guessed the word!")
        guess_entry.config(state="disabled")
        submit_button.config(state="disabled")
    else:
        current_row += 1
        if current_row >= rows:
            messagebox.showinfo("Game Over", f"Better luck next time!\nThe word was: {secret_word.upper()}")

# Bind the submit button and the Return key to the submit_guess function.
submit_button.config(command=submit_guess)
guess_entry.bind("<Return>", submit_guess)

# Start the GUI event loop.
root.mainloop()
