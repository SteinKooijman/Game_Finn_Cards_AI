"""
Script to train the RL AI agent and set it as the global agent.
Run this script first to train the model, then use strategy_rl_ai in simulate.py
"""

from rl_ai import train_and_set_agent

if __name__ == "__main__":
    # Train the agent and set it globally
    # The trained model will be stored in the global variable
    # and can be used by strategy_rl_ai when imported in simulate.py
    
    # Reward parameters (optimized via grid search)
    # Update these values based on grid_search_results.csv if available
    draw_percentage = 0.175  # Percentage multiplier for drawing reward
    loss_percentage = 0.175  # Percentage multiplier for loss penalty
    
    train_and_set_agent(
        num_episodes=1000,
        hidden_layers=5,
        hidden_units=30,
        learning_rate=0.001,
        verbose=True,
        draw_percentage=draw_percentage,
        loss_percentage=loss_percentage
    )
    
    print("\n" + "=" * 70)
    print("Training complete! The trained model is now stored globally and saved to disk.")
    print("The model has been saved to 'best_model.weights.h5'")
    print("You can now run simulate.py and it will use this trained model.")
    print("=" * 70)

