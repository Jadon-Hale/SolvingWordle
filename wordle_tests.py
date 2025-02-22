import unittest
import numpy as np
import os
import torch
import random

from wordleRL import WordleEnv, get_colors

class TestWordleEnvironment(unittest.TestCase):
    def setUp(self):
        self.allowed_words = ["apple", "grape", "peach"]
        # Create an environment instance with 6 attempts.
        self.env = WordleEnv(self.allowed_words, max_attempts=6)
        # fix the random seed.
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

    def test_reset_state(self):

        state = self.env.reset()
        self.assertEqual(state.shape, (60,))
        self.assertTrue((self.env.letters == -1).all())
        self.assertTrue((self.env.colors == -1).all())

    def test_win_condition(self):
        
        self.env.reset()
        self.env.secret = "apple"
        action = self.allowed_words.index("apple")
        state, reward, done, _ = self.env.step(action)

        self.assertTrue(done)
        self.assertGreater(reward, 0)
        
        # Verify that the first row in the letters grid matches "apple".
        expected_letters = np.array([ord(c) - ord('a') for c in "apple"])
        np.testing.assert_array_equal(self.env.letters[0], expected_letters)

    def test_loss_condition(self):

        env = WordleEnv(self.allowed_words, max_attempts=2)
        env.reset()
        env.secret = "apple"
        wrong_guess = "peach"
        
        action = self.allowed_words.index(wrong_guess)

        state, reward1, done, _ = env.step(action)
        self.assertFalse(done)

        state, reward2, done, _ = env.step(action)
        self.assertTrue(done)
        self.assertEqual(reward2, -1.0)

    def test_repeated_guess_penalty(self):
        # Check that guessing the same word twice in a row yields a lower reward on the second guess.
        env = WordleEnv(self.allowed_words, max_attempts=3)
        env.reset()
        env.secret = "apple"
        wrong_guess = "peach"
        action = self.allowed_words.index(wrong_guess)

        state, reward1, done, _ = env.step(action)
        self.assertFalse(done)
        
        # Second guess (same word, should incur repeated guess penalty).
        state, reward2, done, _ = env.step(action)
        self.assertFalse(done)
        self.assertLess(reward2, reward1)

    def test_state_update_after_step(self):
        # After making a guess, verify that the state's letter and color grids are updated correctly.
        self.env.reset()
        self.env.secret = "apple"

        action = self.allowed_words.index("grape")
        state, reward, done, _ = self.env.step(action)
        
        # Check that the first row of letters matches the guessed word "grape".
        expected_letters = np.array([ord(c) - ord('a') for c in "grape"])
        np.testing.assert_array_equal(self.env.letters[0], expected_letters)
        
        # Verify that the colors row corresponds to the feedback computed by get_colors.
        expected_feedback = np.array(get_colors("grape", "apple"))
        np.testing.assert_array_equal(self.env.colors[0], expected_feedback)

    def test_render_terminal(self):

        self.env.reset()
        try:
            self.env.render_terminal()
        except Exception as e:
            self.fail("render_terminal() raised an exception: " + str(e))

    def test_render_colored(self):

        self.env.reset()
        try:
            WordleEnv.render_colored(self.env, pause_time=0.1)
        except Exception as e:
            self.fail("render_colored() raised an exception: " + str(e))

if __name__ == '__main__':
    unittest.main()
