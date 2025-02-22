import pickle
import torch
from wordleRL import WordleEnv, load_word_list, compute_word_embeddings, ActorCritic

allowed_words = load_word_list(r"C:\Users\jadon\Documents\Code\wordle\trainlist.txt")
allowed_words = allowed_words[:100]

env = WordleEnv(allowed_words, max_attempts=6)
state_dim = int(env.observation_space.shape[0])
latent_dim = 130
embeddings = compute_word_embeddings(allowed_words)

with open("saved_agent_100k.pkl", "rb") as f:
    agent = pickle.load(f)

num_eval = 100
wins = 0
total_guesses = 0
render_eval = True

for _ in range(num_eval):
    state = env.reset()
    done = False
    guesses = 0
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        logits, _ = agent(state_tensor)
        probs = torch.softmax(logits, dim=1)

        action = torch.argmax(probs, dim=1).item()
        state, reward, done, _ = env.step(action)
        if render_eval:
            env.render_colored(pause_time=0.5)
        guesses += 1
    if reward > 0:
        wins += 1
    total_guesses += guesses

win_rate = wins / num_eval * 100
avg_guesses = total_guesses / num_eval
print(f"Evaluation over {num_eval} games: Win Rate = {win_rate:.2f}%, Average Guesses = {avg_guesses:.2f}")
