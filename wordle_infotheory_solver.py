import random
import math
import matplotlib.pyplot as plt
import tqdm

# ------------------------------
# Allowed words
def load_word_list(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# ------------------------------
# Feedback function
def get_colors(guess, secret):
    result = [0] * len(guess)
    secret_counts = {}
    for letter in secret:
        secret_counts[letter] = secret_counts.get(letter, 0) + 1

    for i in range(len(guess)):
        if guess[i] == secret[i]:
            result[i] = 2
            secret_counts[guess[i]] -= 1

    for i in range(len(guess)):
        if result[i] == 0:
            if guess[i] in secret_counts and secret_counts[guess[i]]  >0:
                result[i] = 1
                secret_counts[guess[i]] -= 1
    return result

# ------------------------------
# Compute the entropy of a given guess
def compute_entropy(guess, possible_words):
    pattern_counts = {}
    total = len(possible_words)
    for word in possible_words:
        pattern = tuple(get_colors(guess, word))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    entropy = 0.0
    for count in pattern_counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

# ------------------------------
# choose the guess with the highest expected entropy.
def choose_best_guess(possible_words, allowed_words):
    if len(possible_words) == 1:
        return possible_words[0]
    best_guess = None
    best_entropy = -1
    for word in allowed_words:
        ent = compute_entropy(word, possible_words)
        if ent > best_entropy:
            best_entropy = ent
            best_guess = word
    return best_guess

# ------------------------------
# Play one simulated game of Wordle
def play_game(secret_word, allowed_words, display=False):
    possible_words = allowed_words[:]
    guesses = []
    for attempt in range(1, 7):
        guess = choose_best_guess(possible_words, allowed_words)
        guesses.append(guess)
        feedback = tuple(get_colors(guess, secret_word))
        if display:
            print(f"Attempt {attempt}: Guess: {guess.upper()}, Feedback: {feedback}")
        if guess == secret_word:
            return attempt, True, guesses
        # Filter possible words to those that yield the same feedback for this guess.
        possible_words = [w for w in possible_words if tuple(get_colors(guess, w)) == feedback]
        if not possible_words:
            break  # no words left – game failure
    return 7, False, guesses

# ------------------------------
# Simulate many games, collect scores, and output statistics.
def simulate_games(num_games, allowed_words, display=False):
    total_guesses = 0
    wins = 0
    scores = []
    for i in tqdm.tqdm(range(num_games)):
        secret_word = random.choice(allowed_words)
        attempts, solved, guess_list = play_game(secret_word, allowed_words, display=display)
        scores.append(attempts)
        total_guesses += attempts
        if solved:
            wins += 1
        if display:
            result = f"Solved in {attempts} attempts" if solved else "Failed to solve"
            print(f"Game {i+1}: Secret word: {secret_word.upper()}, {result}, Guesses: {guess_list}")
            print("-" * 40)
    avg_attempts = total_guesses / num_games
    win_rate = (wins / num_games) * 100
    print(f"Simulated {num_games} games.")
    print(f"Average attempts: {avg_attempts:.2f}")
    print(f"Win rate: {win_rate:.2f}%")
    return scores

# ------------------------------
# Plot the score distribution as a histogram.
def plot_score_distribution(scores):
    bins = range(1, 9)
    plt.hist(scores, bins=bins, edgecolor="black", align="left")
    plt.xlabel("Number of Guesses (7 = failure)")
    plt.ylabel("Frequency")
    plt.title("Score Distribution of Wordle Games")
    plt.xticks(range(1, 8))
    plt.show()


if __name__ == "__main__":
    WORDS = load_word_list(r"C:\Users\jadon\Documents\Code\wordle\trainlist.txt")[:1000]
    num_games = 100
    scores = simulate_games(num_games, WORDS, display=False)
    plot_score_distribution(scores)
