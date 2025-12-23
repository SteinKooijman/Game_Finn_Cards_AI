"""
Script to train the RL AI agent and set it as the global agent.
Run this script first to train the model, then use strategy_rl_ai in simulate.py
"""

from rl_ai import train_and_set_agent

if __name__ == "__main__":
    # Train the agent and set it globally
    # The trained model will be stored in the global variable
    # and can be used by strategy_rl_ai when imported in simulate.py
    train_and_set_agent(
        num_episodes=3000,
        hidden_layers=5,
        hidden_units=40,
        learning_rate=0.001,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("Training complete! The trained model is now stored globally.")
    print("You can now import and use strategy_rl_ai in simulate.py")
    print("=" * 70)

