"""
Reinforcement Learning AI for Card Game

This module implements a TensorFlow-based RL agent that learns to play the card game
by making optimal draw/retire decisions using the REINFORCE policy gradient algorithm.
"""

import os
import warnings
import numpy as np

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info, warnings, and errors
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging
# Disable TensorFlow progress bars and info messages
tf.autograph.set_verbosity(0)

from tensorflow import keras
from typing import List, Tuple, Optional
from card_simple import PlayingCard


# Constants for normalization
MAX_CARDS_IN_DECK = 52
MAX_HAND_SIZE = 20  # Reasonable upper bound
MAX_SUM_CARD_VALUES = 260  # 20 cards * 13 (max value) = 260
MAX_UNIQUE_SUITS = 4


def encode_state(cards_left: int, current_hand: List[PlayingCard], drawn_cards: set) -> np.ndarray:
    """
    Encode game state into a feature vector (Option A).
    
    Args:
        cards_left: Number of cards remaining in deck
        current_hand: List of PlayingCard objects currently in hand
        drawn_cards: Set of cards drawn in current round (same as current_hand as set)
    
    Returns:
        numpy array of shape (12,) with normalized features:
        - cards_left (normalized 0-1)
        - hand_size (normalized 0-1)
        - sum_of_card_values (normalized 0-1)
        - suit_counts: 4 features (Hearts, Diamonds, Clubs, Spades)
        - ace_suits: 4 binary features (1 if Ace of that suit exists, 0 otherwise)
        - unique_suits_in_hand (normalized 0-1)
    """
    # Initialize feature vector
    features = []
    
    # 1. cards_left (normalized)
    features.append(min(cards_left / MAX_CARDS_IN_DECK, 1.0))
    
    # 2. hand_size (normalized)
    hand_size = len(current_hand)
    features.append(min(hand_size / MAX_HAND_SIZE, 1.0))
    
    # 3. sum_of_card_values (normalized)
    sum_values = sum(card.get_value() for card in current_hand)
    features.append(min(sum_values / MAX_SUM_CARD_VALUES, 1.0))
    
    # 4. suit_counts: 4 features (one for each suit)
    suit_counts = [0, 0, 0, 0]  # Hearts, Diamonds, Clubs, Spades
    for card in current_hand:
        suit_index = PlayingCard.SUITS.index(card.suit)
        suit_counts[suit_index] += 1
    features.extend(suit_counts)
    
    # 5. ace_suits: 4 binary features
    ace_suits = [0, 0, 0, 0]  # Hearts, Diamonds, Clubs, Spades
    for card in current_hand:
        if card.rank == 'Ace':
            suit_index = PlayingCard.SUITS.index(card.suit)
            ace_suits[suit_index] = 1
    features.extend(ace_suits)
    
    # 6. unique_suits_in_hand (normalized)
    unique_suits = len(set(card.suit for card in current_hand))
    features.append(min(unique_suits / MAX_UNIQUE_SUITS, 1.0))
    
    return np.array(features, dtype=np.float32)


class RLAgent:
    """
    Reinforcement Learning Agent using REINFORCE policy gradient algorithm.
    """
    
    def __init__(self, state_dim: int = 12, hidden_layers: int = 3, hidden_units: int = 30, 
                 learning_rate: float = 0.001, gamma: float = 0.99):
        """
        Initialize the RL agent.
        
        Args:
            state_dim: Dimension of state vector (default 13)
            hidden_layers: Number of hidden layers (max 5)
            hidden_units: Number of units per hidden layer (max 30)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
        """
        self.state_dim = state_dim
        self.gamma = gamma
        self.hidden_layers = min(hidden_layers, 5)  # Max 5 layers
        self.hidden_units = min(hidden_units, 30)  # Max 30 units per layer
        
        # Build the policy network
        self.model = self._build_model(learning_rate)
        
        # Storage for training
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def _build_model(self, learning_rate: float) -> keras.Model:
        """Build the policy network with softmax output."""
        model = keras.Sequential()
        
        # Input layer
        model.add(keras.layers.Dense(self.hidden_units, activation='relu', input_shape=(self.state_dim,)))
        
        # Hidden layers (up to 5 total layers including input)
        for _ in range(self.hidden_layers - 1):
            model.add(keras.layers.Dense(self.hidden_units, activation='relu'))
        
        # Output layer: 2 actions (draw, retire) with softmax
        model.add(keras.layers.Dense(2, activation='softmax'))
        
        # Compile model
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
        
        return model
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Get action probabilities for a given state.
        
        Args:
            state: Encoded state vector
        
        Returns:
            Array of shape (2,) with probabilities for [draw, retire]
        """
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        # Use __call__ instead of predict to avoid any progress bars or warnings
        probabilities = self.model(state, training=False)
        return probabilities[0].numpy()
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Sample an action from the policy distribution.
        
        Args:
            state: Encoded state vector
        
        Returns:
            0 for "draw", 1 for "retire"
        """
        probabilities = self.predict(state)
        action = np.random.choice(2, p=probabilities)
        return action
    
    def store_transition(self, state: np.ndarray, action: int, reward: float):
        """
        Store a state-action-reward tuple for training.
        
        Args:
            state: Encoded state vector
            action: Action taken (0=draw, 1=retire)
            reward: Reward received
        """
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
    
    def _compute_discounted_returns(self) -> np.ndarray:
        """
        Compute discounted returns for the current episode.
        
        Returns:
            Array of discounted returns for each step
        """
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return np.array(returns)
    
    def train_step(self, baseline: float = 0.0):
        """
        Perform one training step using REINFORCE algorithm.
        Updates the policy network using collected episode data.
        
        Args:
            baseline: Baseline score to subtract from returns (default 7.0, the average
                     score of drawing 1 card and retiring). This makes the "draw 1 and retire"
                     strategy neutral, encouraging exploration of better strategies.
        """
        if len(self.episode_states) == 0:
            return
        
        # Convert to numpy arrays
        states = np.array(self.episode_states)
        actions = np.array(self.episode_actions)
        returns = self._compute_discounted_returns()
        
        # Subtract baseline from returns to make "draw 1 and retire" strategy neutral
        # This prevents the agent from getting stuck in this suboptimal strategy
        returns = returns - baseline
        
        # Normalize returns (optional, helps with training stability)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        # Compute policy gradient loss
        with tf.GradientTape() as tape:
            # Get action probabilities
            action_probs = self.model(states, training=True)
            
            # Select probabilities for taken actions
            action_indices = tf.range(len(actions))
            selected_probs = tf.gather(action_probs, action_indices, axis=0)
            selected_probs = tf.gather(selected_probs, actions, axis=1)
            
            # Compute log probabilities
            log_probs = tf.math.log(selected_probs + 1e-8)
            
            # REINFORCE loss: negative log likelihood weighted by returns
            loss = -tf.reduce_mean(log_probs * returns)
        
        # Compute gradients and update
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Clear episode data
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def save_model(self, filepath: str):
        """Save the model weights to a file."""
        self.model.save_weights(filepath)
    
    def load_model(self, filepath: str):
        """Load model weights from a file."""
        self.model.load_weights(filepath)


def train_agent(agent: RLAgent, num_episodes: int = 1000, verbose: bool = True):
    """
    Train the RL agent by running episodes and updating the policy.
    
    Args:
        agent: RLAgent instance to train
        num_episodes: Number of training episodes
        verbose: Whether to print training progress
    """
    from card_simple import CardGame
    
    episode_scores = []
    episode_rounds = []
    episode_wins = []
    episode_losses = []
    
    for episode in range(num_episodes):
        game = CardGame()
        episode_score = 0
        rounds_won = 0
        rounds_lost = 0
        
        # Play one game (multiple rounds)
        while len(game.deck) > 0 or game.hand:
            # Check if game should end
            if len(game.deck) == 0 and not game.hand:
                break
            
            # Encode current state
            state = encode_state(len(game.deck), game.hand, set(game.hand))
            
            # Select action (handle edge cases)
            if not game.hand:
                action = 0  # Must draw if hand is empty
            elif len(game.deck) == 0:
                action = 1  # Must retire if deck is empty
            else:
                action = agent.select_action(state)
            
            # Execute action
            if action == 0:  # Draw
                # Calculate potential value before drawing (for improvement calculation)
                current_hand_score = sum(card.get_value() for card in game.hand) * len(game.hand) if game.hand else 0
                
                drawn_card, _ = game.draw_card()
                if drawn_card is None:
                    continue
                
                success, _ = game.add_card_to_hand(drawn_card)
                
                if not success:
                    # Lost due to duplicate suit
                    # Small fixed penalty - losing means 0 points, not negative points
                    # This discourages risky draws without being too harsh
                    reward = -10.0
                    agent.store_transition(state, action, reward)
                    episode_score += 0  # Lost round, no score
                    rounds_lost += 1
                    game.reset_hand()
                    # Continue to next round (don't break)
                    continue
                else:
                    # Successfully added card
                    # Reward based on improvement in potential value
                    # This encourages building larger hands while keeping drawing rewards small
                    new_hand_score = sum(card.get_value() for card in game.hand) * len(game.hand)
                    improvement = new_hand_score - current_hand_score
                    reward = improvement * 0.1  # 10% of improvement - small fraction to keep drawing rewards small
                    agent.store_transition(state, action, reward)
            
            elif action == 1:  # Retire
                if not game.hand:
                    continue
                
                round_score = game.retire_hand()
                # Reward is the actual score - this is the only real source of points
                reward = round_score
                agent.store_transition(state, action, reward)
                episode_score += round_score
                rounds_won += 1
                game.reset_hand()
                # Continue to next round
                continue
        
        # Update policy after each episode
        agent.train_step()
        
        episode_scores.append(episode_score)
        episode_rounds.append(rounds_won + rounds_lost)
        episode_wins.append(rounds_won)
        episode_losses.append(rounds_lost)
        
        if verbose and (episode + 1) % 100 == 0:
            # Calculate metrics for last 100 episodes
            recent_scores = episode_scores[-100:]
            recent_rounds = episode_rounds[-100:]
            recent_wins = episode_wins[-100:]
            recent_losses = episode_losses[-100:]
            
            avg_score = np.mean(recent_scores)
            avg_rounds = np.mean(recent_rounds)
            total_wins = sum(recent_wins)
            total_losses = sum(recent_losses)
            total_rounds = total_wins + total_losses
            win_rate = (total_wins / total_rounds * 100) if total_rounds > 0 else 0
            max_score = max(recent_scores)
            min_score = min(recent_scores)
            
            # Calculate improvement (compare to previous 100 episodes if available)
            improvement = ""
            if len(episode_scores) >= 200:
                prev_avg = np.mean(episode_scores[-200:-100])
                score_diff = avg_score - prev_avg
                improvement = f" | Improvement: {score_diff:+.2f}"
            
            print(f"\nEpisode {episode + 1}/{num_episodes} (Last 100 episodes):")
            print(f"  Average Score: {avg_score:.2f} (Max: {max_score:.0f}, Min: {min_score:.0f}){improvement}")
            print(f"  Win Rate: {win_rate:.1f}% ({total_wins} wins, {total_losses} losses)")
            print(f"  Average Rounds per Game: {avg_rounds:.2f}")


# Global agent instance (can be trained and reused)
_global_agent = None


def get_agent() -> RLAgent:
    """Get or create the global RL agent instance."""
    global _global_agent
    if _global_agent is None:
        _global_agent = RLAgent()
    return _global_agent


def set_global_agent(agent: RLAgent):
    """
    Set the global agent instance to use for strategy_rl_ai.
    
    Args:
        agent: Trained RLAgent instance to use globally
    """
    global _global_agent
    _global_agent = agent


def train_and_set_agent(num_episodes: int = 1000, hidden_layers: int = 3, 
                        hidden_units: int = 30, learning_rate: float = 0.001, 
                        verbose: bool = True) -> RLAgent:
    """
    Train an RL agent and set it as the global agent.
    
    Args:
        num_episodes: Number of training episodes
        hidden_layers: Number of hidden layers (max 5)
        hidden_units: Number of units per hidden layer (max 30)
        learning_rate: Learning rate for optimizer
        verbose: Whether to print training progress
    
    Returns:
        The trained RLAgent instance
    """
    print("Initializing RL Agent...")
    agent = RLAgent(hidden_layers=hidden_layers, hidden_units=hidden_units, 
                   learning_rate=learning_rate)
    
    print(f"Training agent for {num_episodes} episodes...")
    train_agent(agent, num_episodes=num_episodes, verbose=verbose)
    
    # Set as global agent
    set_global_agent(agent)
    
    print("\nTraining complete! Global agent is now set.")
    return agent


def strategy_rl_ai(cards_left: int, current_hand: List[PlayingCard], drawn_cards: set) -> str:
    """
    RL AI strategy function compatible with simulate.py interface.
    
    This function uses the global trained agent. Make sure to call train_and_set_agent()
    or set_global_agent() before using this strategy.
    
    Args:
        cards_left: Number of cards remaining in deck
        current_hand: List of PlayingCard objects currently in hand
        drawn_cards: Set of cards drawn in current round (same as current_hand as set)
    
    Returns:
        "draw" to draw a card, "retire" to retire the hand
    """
    agent = get_agent()  # Gets the global agent (trained or untrained)
    
    # Handle edge cases
    if not current_hand:
        return "draw"  # Cannot retire empty hand
    
    if cards_left == 0:
        return "retire"  # Must retire if deck is empty
    
    # Encode state and get action
    state = encode_state(cards_left, current_hand, drawn_cards)
    action = agent.select_action(state)
    
    return "draw" if action == 0 else "retire"


if __name__ == "__main__":
    # Example: Train the agent and set it globally
    train_and_set_agent(num_episodes=1000, verbose=True)
    
    print("\nYou can now use strategy_rl_ai in simulate.py")
    print("Example: from rl_ai import strategy_rl_ai")
    print("         simulate(100, strategy_rl_ai)")

