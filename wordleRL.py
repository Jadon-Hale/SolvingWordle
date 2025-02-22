import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tqdm
import pickle

#####################################
# Utility Functions
#####################################

def load_word_list(file_path):
    """Load a list of words from a file (one per line)."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def get_colors(guess, secret):
    """
    Compute feedback for a guess compared to the secret word.
    Returns a list of integers for each letter:
      2 = green (correct position)
      1 = yellow (present but wrong position)
      0 = gray (not in word)
    Handles duplicate letters appropriately.
    """
    result = [0] * len(guess)
    secret_counts = {}
    for letter in secret:
        secret_counts[letter] = secret_counts.get(letter, 0) + 1

    # First pass: mark greens.
    for i in range(len(guess)):
        if guess[i] == secret[i]:
            result[i] = 2
            secret_counts[guess[i]] -= 1

    # Second pass: mark yellows.
    for i in range(len(guess)):
        if result[i] == 0 and guess[i] in secret_counts and secret_counts[guess[i]] > 0:
            result[i] = 1
            secret_counts[guess[i]] -= 1

    return result

def compute_word_embeddings(allowed_words):
    """
    Precompute a fixed embedding matrix for each allowed word.
    Each word is represented as a 130-dimensional vector (5 letters * 26 one-hot entries).
    Returns a torch.Tensor of shape (num_words, 130).
    """
    embeddings = []
    for word in allowed_words:
        vec = []
        for letter in word:
            one_hot = [0] * 26
            idx = ord(letter) - ord('a')
            one_hot[idx] = 1
            vec.extend(one_hot)
        embeddings.append(vec)
    embeddings = np.array(embeddings, dtype=np.float32)
    return torch.tensor(embeddings)  # shape: (num_words, 130)

def is_consistent(current_guess, prev_guess, prev_feedback):
    """
    Checks whether the current guess is consistent with a previous guess and its feedback.
    For each letter position in the previous guess:
      - If feedback was green (2), then the current guess must have the same letter in that position.
      - If feedback was yellow (1), then the letter must appear in the current guess but in a different position.
      - If feedback was gray (0) and that letter was never marked yellow/green in the same guess,
        then the current guess should not include that letter.
    Returns True if consistent, False otherwise.
    """
    for i in range(len(prev_guess)):
        letter = prev_guess[i]
        fb = prev_feedback[i]
        if fb == 2:
            # Must match exactly.
            if current_guess[i] != letter:
                return False
        elif fb == 1:
            # Must appear elsewhere.
            if letter not in current_guess or current_guess[i] == letter:
                return False
        elif fb == 0:
            # If this letter never got a positive signal in this guess, then it shouldn't appear.
            # (Note: This is a simplification; sometimes gray letters can appear if they are duplicates.)
            if letter not in [prev_guess[j] for j in range(len(prev_guess)) if prev_feedback[j] > 0]:
                if letter in current_guess:
                    return False
    return True

#####################################
# Custom Reward Function
#####################################

def compute_custom_reward(secret, current_guess, previous_feedbacks, previous_guesses):
    """
    Compute a reward for the current guess that now incorporates:
      - A heavy penalty for guessing a word already tried.
      - Per-letter rewards/penalties (greens, yellows, grays) with bonuses for improvements.
      - Consistency checks: penalize if the guess is inconsistent with prior feedback;
        bonus if it is consistent.
      - Overall improvement bonus based on total greens.
      
    Parameters:
      secret (str): The secret word.
      current_guess (str): The current guess word.
      previous_feedbacks (list of lists): List of feedback arrays (each length 5) from previous guesses.
      previous_guesses (list of str): List of previous guess words.
      
    Returns:
      reward (float): The computed reward.
      current_feedback (list): Feedback for the current guess.
    """
    reward = 0.0

    # 1. Repeated Guess Penalty.
    if current_guess in previous_guesses:
        reward -= 3.0  # Heavy penalty for repeating a guess.

    # 2. Compute current feedback.
    current_feedback = get_colors(current_guess, secret)
    
    # 3. Per-letter improvement reward (as before).
    prev_best = [-1] * len(current_feedback)
    if previous_feedbacks:
        for pos in range(len(current_feedback)):
            prev_best[pos] = max(feedback[pos] for feedback in previous_feedbacks)
    for pos in range(len(current_feedback)):
        curr_fb = current_feedback[pos]
        prev_fb = prev_best[pos]
        if curr_fb == 2:  # Green
            reward += 1.0
            if prev_fb < 2:
                reward += 0.5  # Bonus for improvement.
        elif curr_fb == 1:  # Yellow
            reward += 0.5
            if prev_fb < 1:
                reward += 0.2  # Bonus if improving from gray.
            if prev_fb == 2:
                reward -= 1.0  # Heavy penalty for downgrading from green.
        else:  # Gray
            reward -= 0.2
            if prev_fb >= 1:
                reward -= 0.5  # Extra penalty if this letter was known to be in the word.

    # 4. Consistency Check across previous guesses.
    consistency_penalty = 0.0
    consistency_bonus = 0.0
    if previous_guesses and previous_feedbacks:
        for pg, pf in zip(previous_guesses, previous_feedbacks):
            if not is_consistent(current_guess, pg, pf):
                consistency_penalty += 1.0  # Penalize for inconsistency.
            else:
                consistency_bonus += 0.5  # Bonus for being consistent.
    reward -= consistency_penalty
    reward += consistency_bonus

    # 5. Overall improvement bonus based on total greens.
    current_total_greens = current_feedback.count(2)
    if previous_feedbacks:
        avg_prev_greens = sum(feedback.count(2) for feedback in previous_feedbacks) / len(previous_feedbacks)
        if current_total_greens > avg_prev_greens:
            reward += (current_total_greens - avg_prev_greens) * 0.5
        elif current_total_greens < avg_prev_greens:
            reward -= (avg_prev_greens - current_total_greens) * 0.5

    return reward, current_feedback

#####################################
# Modified Wordle Environment with Custom Reward
#####################################

class WordleEnv(gym.Env):
    """
    A Gym-style environment for Wordle.
    State: Two grids (6x5 for letters and 6x5 for feedback), flattened.
    Action: An integer index selecting a 5-letter guess from the allowed words.
    """
    def __init__(self, allowed_words, max_attempts=6):
        super(WordleEnv, self).__init__()
        self.allowed_words = allowed_words
        self.max_attempts = max_attempts
        self.action_space = gym.spaces.Discrete(len(allowed_words))
        self.observation_space = gym.spaces.Box(low=-1, high=26, shape=(self.max_attempts, 5, 2), dtype=np.int32)
        self.reset()

    def reset(self):
        self.secret = random.choice(self.allowed_words)
        self.letters = -1 * np.ones((self.max_attempts, 5), dtype=np.int32)
        self.colors = -1 * np.ones((self.max_attempts, 5), dtype=np.int32)
        self.current_attempt = 0
        self.previous_feedbacks = []  # Reset history of feedback.
        self.previous_guesses = []
        return self._get_state()

    def _get_state(self):
        state = np.concatenate([self.letters.flatten(), self.colors.flatten()]).astype(np.float32)
        return state
    
    def step(self, action):
        guess = self.allowed_words[action]
        guess_indices = [ord(c) - ord('a') for c in guess]
        self.letters[self.current_attempt, :] = np.array(guess_indices)
        
        # Use the improved reward function.
        reward, current_feedback = compute_custom_reward(
            self.secret, guess, self.previous_feedbacks, self.previous_guesses
        )
        self.colors[self.current_attempt, :] = np.array(current_feedback)
        
        # Record the feedback and the guess.
        self.previous_feedbacks.append(current_feedback)
        self.previous_guesses.append(guess)
        
        done = False
        if guess == self.secret:
            reward += 1.0  # Additional win bonus.
            done = True
        
        self.current_attempt += 1
        if self.current_attempt >= self.max_attempts and guess != self.secret:
            done = True
            reward = -1.0  # Loss penalty.
        
        next_state = self._get_state()
        return next_state, reward, done, {}

    def render_terminal(self, mode='human'):
        for i in range(self.max_attempts):
            row_letters = ''.join([chr(x + ord('a')) if x >= 0 else '_' for x in self.letters[i]])
            row_colors = ''.join([str(c) if c >= 0 else '_' for c in self.colors[i]])
            print(f"{row_letters}   {row_colors}")
        print()

    def render_colored(env, pause_time=0.5):
        # (Rendering code omitted for brevity; remains unchanged.)
        fig, ax = plt.subplots(figsize=(5,6))
        ax.set_xlim(0,5)
        ax.set_ylim(0,6)
        ax.invert_yaxis()
        ax.axis('off')
        for i in range(env.max_attempts):
            for j in range(5):
                letter = env.letters[i, j]
                feedback = env.colors[i, j]
                if feedback == 2:
                    color = "#6aaa64"
                elif feedback == 1:
                    color = "#c9b458"
                elif feedback == 0:
                    color = "#787c7e"
                else:
                    color = "white"
                rect = patches.Rectangle((j, i), 1, 1, edgecolor='black', facecolor=color)
                ax.add_patch(rect)
                if letter != -1:
                    char = chr(letter + ord('a')).upper()
                    ax.text(j + 0.5, i + 0.5, char, ha='center', va='center', fontsize=20, weight='bold', color='black')
        plt.draw()
        plt.pause(pause_time)
        plt.close(fig)

#####################################
# Policy Network with Output Reduction
#####################################
    
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, latent_dim, word_embeddings):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, latent_dim)
        # Add a LayerNorm for the latent dimension.
        self.ln = nn.LayerNorm(latent_dim)
        self.word_embeddings = word_embeddings  # Fixed tensor of shape (num_actions, latent_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        latent = self.fc2(x)
        # Apply layer normalization.
        latent = self.ln(latent)
        logits = torch.matmul(latent, self.word_embeddings.t())
        return logits
    
class ActorCritic(nn.Module):
    def __init__(self, state_dim, latent_dim, word_embeddings):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)
        # Actor
        self.actor_word_embeddings = word_embeddings  # shape: (num_actions, latent_dim)
        # Critic
        self.critic_head = nn.Linear(latent_dim, 1)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        latent = self.fc2(x)
        latent = self.ln(latent)
        # Actor output: compute logits via dot product with word embeddings.
        logits = torch.matmul(latent, self.actor_word_embeddings.t())
        # Critic output: a single scalar.
        value = self.critic_head(latent)
        return logits, value

#####################################
# Training Loop (REINFORCE) with averaged gradients
#####################################

def train_agent_avg_gradients(env, policy_net, optimizer, num_episodes, gamma=0.99, render=False):
    torch.autograd.set_detect_anomaly(True)
    episode_rewards = []
    print("Training Agent with Average Gradient Updates:")
    for episode in tqdm.tqdm(range(num_episodes)):
        state = env.reset()
        states = []
        actions = []
        rewards = []
        done = False

        # Collect trajectory for one episode.
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = policy_net(state_tensor)  # Shape: (1, num_actions)
            probs = torch.softmax(logits, dim=1)
            m = Categorical(probs)
            action = m.sample()
            next_state, reward, done, _ = env.step(action.item())
            
            if render:
                env.render_colored()
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        # Compute discounted returns.
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Compute gradient for each guess and accumulate.
        avg_gradients = None
        num_steps = len(states)
        for s, a, R in zip(states, actions, returns):
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            logits = policy_net(s_tensor)
            probs = torch.softmax(logits, dim=1)
            m = Categorical(probs)
            loss = - m.log_prob(a) * R
            
            # Compute gradients for this step.
            grad_list = torch.autograd.grad(loss, policy_net.parameters(), retain_graph=True)
            if avg_gradients is None:
                avg_gradients = [g.clone() for g in grad_list]
            else:
                for i, g in enumerate(grad_list):
                    avg_gradients[i] += g
        
        # Average the gradients.
        for i in range(len(avg_gradients)):
            avg_gradients[i] /= num_steps
        
        # Zero out any existing gradients and manually assign our averaged gradients.
        optimizer.zero_grad()
        for param, grad in zip(policy_net.parameters(), avg_gradients):
            param.grad = grad
        optimizer.step()

        episode_total_reward = sum(rewards)
        episode_rewards.append(episode_total_reward)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {episode_total_reward:.2f}")
    return episode_rewards

#####################################
# Training Loop (A2C)
#####################################

def train_agent_A2C(env, ac_net, optimizer, num_episodes, gamma=0.99, render=False):
    torch.autograd.set_detect_anomaly(True)
    episode_rewards = []
    print("Training Agent (A2C with TD targets and epsilon-greedy):")
    
    # Epsilon parameters.
    epsilon = 1.0          # start with full exploration.
    epsilon_decay = 0.995  # decay factor per episode.
    
    for episode in tqdm.tqdm(range(num_episodes)):
        state = env.reset()
        states, actions, rewards, values, td_targets = [], [], [], [], []
        done = False
        
        # Collect trajectory for one episode.
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, value = ac_net(state_tensor)
            probs = torch.softmax(logits, dim=1)
            m = Categorical(probs)
            
            # Epsilon-greedy action selection:
            if random.random() < epsilon:
                action = env.action_space.sample()
                action = torch.tensor(action)
            else:
                action = m.sample()
            
            next_state, reward, done, _ = env.step(action.item())
            if render:
                env.render_colored()
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value.item())
            
            # Compute TD target: if done, V(s') = 0.
            if done:
                next_value = 0.0
            else:
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                _, next_value = ac_net(next_state_tensor)
                next_value = next_value.item()
            td_target = reward + gamma * next_value
            td_targets.append(td_target)
            
            state = next_state
        
        epsilon = epsilon * epsilon_decay
        
        # Compute advantages using TD targets.
        advantages = [td - v for td, v in zip(td_targets, values)]
        # Convert lists to tensors.
        advantages = torch.tensor(advantages, dtype=torch.float32)
        td_targets = torch.tensor(td_targets, dtype=torch.float32)
        
        # Compute losses.
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        
        for s, a, adv, td in zip(states, actions, advantages, td_targets):
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            logits, value = ac_net(s_tensor)
            probs = torch.softmax(logits, dim=1)
            m = Categorical(probs)
            log_prob = m.log_prob(a)
            
            # Policy loss weighted by advantage.
            actor_loss -= log_prob * adv
            # Critic loss: mean squared error between TD target and predicted value.
            critic_loss += (td - value) ** 2
            # Entropy bonus.
            entropy_loss -= m.entropy()
        
        total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        episode_total_reward = sum(rewards)
        episode_rewards.append(episode_total_reward)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {episode_total_reward:.2f}\n")
    
    return episode_rewards

#####################################
# Main Training and Evaluation
#####################################

if __name__ == "__main__":
    # Load allowed words from a file (one word per line).
    allowed_words = load_word_list(r"C:\Users\jadon\Documents\Code\wordle\trainlist.txt")
    # For curriculum learning, you might start with a subset.
    #random.shuffle(allowed_words)
    allowed_words = allowed_words[:100]

    # Create the environment.
    env = WordleEnv(allowed_words)
    state_dim = int(np.prod(env.observation_space.shape))  # e.g., 6*5*2 = 60
    latent_dim = 130  # 5 letters x 26 one-hot dimensions.
    word_embeddings = compute_word_embeddings(allowed_words)  # Tensor of shape (num_actions, 130)
    
    a2c=True

    if a2c:
        ac_net = ActorCritic(state_dim, latent_dim, word_embeddings)
        optimizer = optim.Adam(ac_net.parameters(), lr=1e-5)
        
        num_training_episodes = 100000
        rewards = train_agent_A2C(env, ac_net, optimizer, num_training_episodes, render=False)
        
        # Evaluate
        num_eval = 100
        wins = 0
        total_guesses = 0
        render_eval = False
        print("Evaluating Agent: ")
        for _ in tqdm.tqdm(range(num_eval)):
            state = env.reset()
            done = False
            guesses = 0
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                logits, value = ac_net(state_tensor)
                print("Value: ", value)
                probs = torch.softmax(logits, dim=1)
                action = torch.argmax(probs, dim=1).item()
                state, reward, done, _ = env.step(action)
                
                if render_eval:
                    env.render_colored()
                
                guesses += 1
            if reward > 0:
                wins += 1
            total_guesses += guesses
            
        with open("saved_agent_100k.pkl", "wb") as f:
            pickle.dump(ac_net, f)

        print("Agent saved as 'saved_agent.pkl'.")

        win_rate = wins / num_eval * 100
        avg_guesses = total_guesses / num_eval
        print(f"Evaluation over {num_eval} games: Win Rate = {win_rate:.2f}%, Average Guesses = {avg_guesses:.2f}")

        # Plot training rewards.
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Rewards over Episodes")
        plt.show()
    
    else:
        policy_net = PolicyNetwork(state_dim, latent_dim, word_embeddings)
        optimizer = optim.Adam(policy_net.parameters(), lr=1e-5)
        
        num_training_episodes = 1000
        rewards = train_agent_avg_gradients(env, policy_net, optimizer, num_training_episodes)

        # Evaluate
        num_eval = 100
        wins = 0
        total_guesses = 0
        render_eval = True
        print("Evaluating Agent: ")
        for _ in tqdm.tqdm(range(num_eval)):
            state = env.reset()
            done = False
            guesses = 0
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                logits = policy_net(state_tensor)
                probs = torch.softmax(logits, dim=1)
                action = torch.argmax(probs, dim=1).item()
                state, reward, done, _ = env.step(action)
                
                if render_eval:
                    env.render_colored()
                
                guesses += 1
            if reward > 0:
                wins += 1
            total_guesses += guesses

        win_rate = wins / num_eval * 100
        avg_guesses = total_guesses / num_eval
        print(f"Evaluation over {num_eval} games: Win Rate = {win_rate:.2f}%, Average Guesses = {avg_guesses:.2f}")

        # Plot training rewards.
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Rewards over Episodes")
        plt.show()