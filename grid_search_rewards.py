"""
Grid Search Script for Reward Parameters

This script performs a 2D grid search over draw_percentage and loss_percentage parameters,
trains models for each combination, evaluates them, and saves the best performing model.
"""

import os
import warnings
import numpy as np

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

from rl_ai import RLAgent, train_agent, encode_state
from card_simple import CardGame

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Training parameters
TRAINING_EPISODES = 300  # Number of training episodes per combination

# Neural network architecture (same for all combinations)
HIDDEN_LAYERS = 5
HIDDEN_UNITS = 30
LEARNING_RATE = 0.001

# Grid search parameters
GRID_MIN_PERCENTAGE = 0.05  # 5%
GRID_MAX_PERCENTAGE = 0.3  # 50%
GRID_STEPS = 3  # 10 steps = 100 combinations (10x10)

# Evaluation parameters
EVALUATION_GAMES = 50  # Number of games to evaluate each trained model

# Model saving
BEST_MODEL_SAVE_PATH = "best_model.weights.h5"
RESULTS_CSV_PATH = "grid_search_results.csv"

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_agent(agent: RLAgent, num_games: int = 50) -> dict:
    """
    Evaluate a trained agent by running multiple games and returning statistics.
    
    Args:
        agent: Trained RLAgent instance
        num_games: Number of evaluation games to run
    
    Returns:
        Dictionary with keys: 'mean', 'std', 'min', 'max' containing evaluation statistics
    """
    scores = []
    
    for _ in range(num_games):
        game = CardGame()
        game_score = 0
        
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
                drawn_card, _ = game.draw_card()
                if drawn_card is None:
                    continue
                
                success, _ = game.add_card_to_hand(drawn_card)
                
                if not success:
                    # Lost due to duplicate suit
                    # Calculate score of hand before the losing card
                    hand_before_loss = game.hand[:-1] if len(game.hand) > 1 else []
                    if hand_before_loss:
                        total_value = sum(card.get_value() for card in hand_before_loss)
                        num_cards = len(hand_before_loss)
                        round_score = total_value * num_cards
                    else:
                        round_score = 0
                    game_score += round_score
                    game.reset_hand()
                    continue
                # If successful, continue loop (card is now in hand)
            
            elif action == 1:  # Retire
                if not game.hand:
                    continue
                
                round_score = game.retire_hand()
                game_score += round_score
                game.reset_hand()
                continue
        
        scores.append(game_score)
    
    scores_array = np.array(scores)
    
    return {
        'mean': np.mean(scores_array),
        'std': np.std(scores_array),
        'min': np.min(scores_array),
        'max': np.max(scores_array)
    }

# ============================================================================
# GRID SEARCH
# ============================================================================

def run_grid_search():
    """
    Run grid search over draw_percentage and loss_percentage parameters.
    """
    # Generate parameter combinations
    percentages = np.linspace(GRID_MIN_PERCENTAGE, GRID_MAX_PERCENTAGE, GRID_STEPS)
    
    results = []
    best_score = float('-inf')
    best_params = None
    best_agent = None
    best_result_stats = None
    
    total_combinations = GRID_STEPS * GRID_STEPS
    current_combination = 0
    
    print("=" * 80)
    print("GRID SEARCH FOR REWARD PARAMETERS")
    print("=" * 80)
    print(f"Training episodes per combination: {TRAINING_EPISODES}")
    print(f"Neural network: {HIDDEN_LAYERS} layers, {HIDDEN_UNITS} units, lr={LEARNING_RATE}")
    print(f"Grid search: {GRID_STEPS}x{GRID_STEPS} = {total_combinations} combinations")
    print(f"Evaluation games per model: {EVALUATION_GAMES}")
    print(f"Parameter range: {GRID_MIN_PERCENTAGE*100:.0f}% to {GRID_MAX_PERCENTAGE*100:.0f}%")
    print("=" * 80)
    print()
    
    for draw_pct in percentages:
        for loss_pct in percentages:
            current_combination += 1
            draw_pct_rounded = round(draw_pct, 3)
            loss_pct_rounded = round(loss_pct, 3)
            
            print(f"[{current_combination}/{total_combinations}] Training with draw_pct={draw_pct_rounded:.1%}, loss_pct={loss_pct_rounded:.1%}...")
            
            # Create new agent with same architecture
            agent = RLAgent(
                hidden_layers=HIDDEN_LAYERS,
                hidden_units=HIDDEN_UNITS,
                learning_rate=LEARNING_RATE
            )
            
            # Train agent
            train_agent(
                agent=agent,
                num_episodes=TRAINING_EPISODES,
                verbose=False,  # Disable verbose output during grid search
                draw_percentage=draw_pct,
                loss_percentage=loss_pct
            )
            
            # Evaluate agent
            print(f"  Evaluating with {EVALUATION_GAMES} games...", end=" ", flush=True)
            eval_stats = evaluate_agent(agent, num_games=EVALUATION_GAMES)
            avg_score = eval_stats['mean']
            print(f"Avg: {avg_score:.2f}, Std: {eval_stats['std']:.2f}, Min: {eval_stats['min']:.0f}, Max: {eval_stats['max']:.0f}")
            
            # Store results
            results.append({
                'draw_percentage': draw_pct_rounded,
                'loss_percentage': loss_pct_rounded,
                'average_score': avg_score,
                'std_score': eval_stats['std'],
                'min_score': eval_stats['min'],
                'max_score': eval_stats['max']
            })
            
            # Track best model (based on average score)
            if avg_score > best_score:
                best_score = avg_score
                best_params = (draw_pct_rounded, loss_pct_rounded)
                best_agent = agent
                best_result_stats = eval_stats
    
    # Print all results
    print()
    print("=" * 110)
    print("GRID SEARCH RESULTS")
    print("=" * 110)
    print(f"{'Draw %':<10} {'Loss %':<10} {'Avg Score':<12} {'Std Dev':<12} {'Min':<10} {'Max':<10}")
    print("-" * 110)
    
    # Sort results by score (descending)
    results_sorted = sorted(results, key=lambda x: x['average_score'], reverse=True)
    
    for result in results_sorted:
        print(f"{result['draw_percentage']:<10.1%} {result['loss_percentage']:<10.1%} "
              f"{result['average_score']:<12.2f} {result['std_score']:<12.2f} "
              f"{result['min_score']:<10.0f} {result['max_score']:<10.0f}")
    
    print()
    print("=" * 80)
    print("BEST MODEL")
    print("=" * 80)
    print(f"Draw Percentage: {best_params[0]:.1%}")
    print(f"Loss Percentage: {best_params[1]:.1%}")
    print(f"Average Score: {best_score:.2f}")
    if best_result_stats:
        print(f"Standard Deviation: {best_result_stats['std']:.2f}")
        print(f"Min Score: {best_result_stats['min']:.0f}")
        print(f"Max Score: {best_result_stats['max']:.0f}")
    print()
    
    # Save best model
    if best_agent is not None:
        print(f"Saving best model to {BEST_MODEL_SAVE_PATH}...")
        best_agent.save_model(BEST_MODEL_SAVE_PATH)
        print("Best model saved!")
        
        # Set as global agent so it can be used by strategy_rl_ai
        from rl_ai import set_global_agent
        set_global_agent(best_agent)
        print("Best model set as global agent!")
    
    # Save results to CSV
    try:
        import csv
        with open(RESULTS_CSV_PATH, 'w', newline='') as csvfile:
            fieldnames = ['draw_percentage', 'loss_percentage', 'average_score', 
                         'std_score', 'min_score', 'max_score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results_sorted:
                writer.writerow(result)
        print(f"Results saved to {RESULTS_CSV_PATH}")
    except Exception as e:
        print(f"Warning: Could not save CSV results: {e}")
    
    print()
    print("=" * 80)
    print("GRID SEARCH COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_grid_search()

