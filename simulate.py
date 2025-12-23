"""
Simulation module for card game strategies.

This module provides functions to simulate multiple games with different strategies
and track detailed statistics.
"""

import random
from card_simple import CardGame, PlayingCard

# Import RL AI strategy (will use global trained model if available)
try:
    from rl_ai import strategy_rl_ai
    _rl_ai_available = True
except ImportError:
    # If rl_ai module is not available, strategy_AI will handle it
    strategy_rl_ai = None
    _rl_ai_available = False


def strategy_50_50(cards_left, current_hand, drawn_cards):
    """
    50/50 strategy: 50% chance to draw, 50% chance to retire.
    
    Args:
        cards_left: Number of cards remaining in deck
        current_hand: List of PlayingCard objects currently in hand
        drawn_cards: Set of cards drawn in current round (same as current_hand as set)
    
    Returns:
        "draw" to draw a card, "retire" to retire the hand
    """
    # Cannot retire empty hand
    if not current_hand:
        return "draw"
    
    # Must retire if deck is empty
    if cards_left == 0:
        return "retire"
    
    # 50/50 chance to draw or retire
    return random.choice(["draw", "retire"])


def strategy_draw_until_one(cards_left, current_hand, drawn_cards):
    """
    Strategy that draws until there is one card in the hand, then always retires.
    
    Args:
        cards_left: Number of cards remaining in deck
        current_hand: List of PlayingCard objects currently in hand
        drawn_cards: Set of cards drawn in current round (same as current_hand as set)
    
    Returns:
        "draw" to draw a card, "retire" to retire the hand
    """
    # Cannot retire empty hand
    if not current_hand:
        return "draw"
    
    # Must retire if deck is empty
    if cards_left == 0:
        return "retire"
    
    # Draw until we have 1 card, then always retire
    if len(current_hand) < 1:
        return "draw"
    else:
        return "retire"


def strategy_draw_until_two(cards_left, current_hand, drawn_cards):
    """
    Strategy that draws until there are two cards in the hand, then always retires.
    
    Args:
        cards_left: Number of cards remaining in deck
        current_hand: List of PlayingCard objects currently in hand
        drawn_cards: Set of cards drawn in current round (same as current_hand as set)
    
    Returns:
        "draw" to draw a card, "retire" to retire the hand
    """
    # Cannot retire empty hand
    if not current_hand:
        return "draw"
    
    # Must retire if deck is empty
    if cards_left == 0:
        return "retire"
    
    # Draw until we have 2 cards, then always retire
    if len(current_hand) < 2:
        return "draw"
    else:
        return "retire"


def strategy_draw_until_three(cards_left, current_hand, drawn_cards):
    """
    Strategy that draws until there are three cards in the hand, then always retires.
    
    Args:
        cards_left: Number of cards remaining in deck
        current_hand: List of PlayingCard objects currently in hand
        drawn_cards: Set of cards drawn in current round (same as current_hand as set)
    
    Returns:
        "draw" to draw a card, "retire" to retire the hand
    """
    # Cannot retire empty hand
    if not current_hand:
        return "draw"
    
    # Must retire if deck is empty
    if cards_left == 0:
        return "retire"
    
    # Draw until we have 3 cards, then always retire
    if len(current_hand) < 3:
        return "draw"
    else:
        return "retire"


def strategy_draw_until_four(cards_left, current_hand, drawn_cards):
    """
    Strategy that draws until there are four cards in the hand, then always retires.
    
    Args:
        cards_left: Number of cards remaining in deck
        current_hand: List of PlayingCard objects currently in hand
        drawn_cards: Set of cards drawn in current round (same as current_hand as set)
    
    Returns:
        "draw" to draw a card, "retire" to retire the hand
    """
    # Cannot retire empty hand
    if not current_hand:
        return "draw"
    
    # Must retire if deck is empty
    if cards_left == 0:
        return "retire"
    
    # Draw until we have 4 cards, then always retire
    if len(current_hand) < 4:
        return "draw"
    else:
        return "retire"


def strategy_AI(cards_left, current_hand, drawn_cards):
    """
    RL AI strategy that uses the trained global model.
    
    IMPORTANT: Train the model first by running train_rl_ai.py
    This strategy uses the globally stored trained model from rl_ai module.
    
    Args:
        cards_left: Number of cards remaining in deck
        current_hand: List of PlayingCard objects currently in hand
        drawn_cards: Set of cards drawn in current round (same as current_hand as set)
    
    Returns:
        "draw" to draw a card, "retire" to retire the hand
    """
    if not _rl_ai_available or strategy_rl_ai is None:
        raise ImportError(
            "rl_ai module not available or not trained. "
            "Please ensure rl_ai.py is in the same directory and run train_rl_ai.py first."
        )
    
    # Use the trained global model
    return strategy_rl_ai(cards_left, current_hand, drawn_cards)


def simulate(num_games, strategy, verbose=True):
    """
    Simulate multiple games using a given strategy.
    
    Args:
        num_games: Number of games to simulate
        strategy: Strategy function that takes (cards_left, current_hand, drawn_cards)
                  and returns "draw" or "retire"
        verbose: If True, print per-game scores. If False, suppress per-game output.
    
    Returns:
        Dictionary containing detailed statistics for all games and aggregate stats
    """
    games_data = []
    
    for game_num in range(1, num_games + 1):
        game = CardGame()
        round_number = 1
        game_rounds = []
        
        # Game loop
        while len(game.deck) > 0 or game.hand:
            # Check if game should end (empty deck and empty hand)
            if len(game.deck) == 0 and not game.hand:
                break
            
            # Get strategy decision
            try:
                decision = strategy(len(game.deck), game.hand, set(game.hand))
            except Exception as e:
                # Handle strategy errors gracefully
                # Default to draw if deck has cards, retire if deck is empty
                if len(game.deck) > 0:
                    decision = "draw"
                else:
                    decision = "retire"
            
            # Validate decision and correct if invalid
            if decision == "retire" and not game.hand:
                # Cannot retire empty hand, must draw if possible
                if len(game.deck) > 0:
                    decision = "draw"
                else:
                    # Deck empty and hand empty - game over
                    break
            elif decision == "draw" and len(game.deck) == 0:
                # Cannot draw from empty deck, must retire if hand exists
                if game.hand:
                    decision = "retire"
                else:
                    # Deck empty and hand empty - game over
                    break
            
            # Execute decision
            if decision == "draw":
                # Draw a card
                drawn_card, _ = game.draw_card()
                
                if drawn_card is None:
                    # Deck is empty (shouldn't happen due to validation above)
                    continue
                
                # Add card to hand
                success, _ = game.add_card_to_hand(drawn_card)
                
                if not success:
                    # Lost due to duplicate suit
                    # Calculate score of hand BEFORE the card that caused the duplicate was added
                    # (the card is already in hand, so we use hand[:-1] to exclude it)
                    hand_before_loss = game.hand[:-1] if len(game.hand) > 1 else []
                    if hand_before_loss:
                        total_value = sum(card.get_value() for card in hand_before_loss)
                        num_cards = len(hand_before_loss)
                        round_score = total_value * num_cards
                    else:
                        round_score = 0
                    game.add_to_total_score(round_score)
                    game_rounds.append({
                        "round": round_number,
                        "score": round_score,
                        "outcome": "lost",
                        "cards_in_hand": len(game.hand)
                    })
                    game.reset_hand()
                    round_number += 1
                    # Continue to next round
                    continue
                # If successful, continue loop (card is now in hand)
                
            elif decision == "retire":
                # Retire the hand
                if not game.hand:
                    # Shouldn't happen due to validation, but check anyway
                    continue
                
                round_score = game.retire_hand()
                cards_in_hand = len(game.hand)
                game.add_to_total_score(round_score)
                game_rounds.append({
                    "round": round_number,
                    "score": round_score,
                    "outcome": "retired",
                    "cards_in_hand": cards_in_hand
                })
                game.reset_hand()
                round_number += 1
                # Continue to next round
        
        # Record game statistics
        final_score = game.total_score
        total_rounds = len(game_rounds)
        
        games_data.append({
            "game_num": game_num,
            "final_score": final_score,
            "rounds": game_rounds,
            "total_rounds": total_rounds
        })
        
        # Print final score for this game (if verbose)
        if verbose:
            print(f"Game {game_num}: Final Score = {final_score}")
    
    # Calculate aggregate statistics
    if games_data:
        final_scores = [g["final_score"] for g in games_data]
        total_rounds_list = [g["total_rounds"] for g in games_data]
        
        # Count wins/losses per round
        round_outcomes = []
        for game in games_data:
            for round_data in game["rounds"]:
                round_outcomes.append(round_data["outcome"])
        
        wins = round_outcomes.count("retired")
        losses = round_outcomes.count("lost")
        total_rounds_played = len(round_outcomes)
        
        # Score distribution
        score_distribution = {}
        for game in games_data:
            for round_data in game["rounds"]:
                score = round_data["score"]
                score_distribution[score] = score_distribution.get(score, 0) + 1
        
        # Calculate standard deviation of final scores
        avg_final_score = sum(final_scores) / len(final_scores)
        if len(final_scores) > 1:
            variance = sum((score - avg_final_score) ** 2 for score in final_scores) / len(final_scores)
            std_dev_final_score = variance ** 0.5
        else:
            std_dev_final_score = 0.0
        
        aggregate = {
            "avg_final_score": avg_final_score,
            "std_dev_final_score": std_dev_final_score,
            "avg_rounds": sum(total_rounds_list) / len(total_rounds_list),
            "total_games": num_games,
            "total_rounds_played": total_rounds_played,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total_rounds_played if total_rounds_played > 0 else 0,
            "score_distribution": score_distribution,
            "min_final_score": min(final_scores),
            "max_final_score": max(final_scores)
        }
    else:
        aggregate = {
            "avg_final_score": 0,
            "std_dev_final_score": 0.0,
            "avg_rounds": 0,
            "total_games": 0,
            "total_rounds_played": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "score_distribution": {},
            "min_final_score": 0,
            "max_final_score": 0
        }
    
    # Return full statistics
    return {
        "games": games_data,
        "aggregate": aggregate
    }


def compare_strategies(num_games, strategies_dict):
    """
    Compare multiple strategies by running simulations and displaying results in a table.
    
    Args:
        num_games: Number of games to simulate for each strategy
        strategies_dict: Dictionary mapping strategy names to strategy functions
                        e.g., {"50/50": strategy_50_50, "Draw Until 2": strategy_draw_until_two}
    
    Returns:
        Dictionary mapping strategy names to their statistics
    """
    results = {}
    
    print(f"Running {num_games} games for each strategy...")
    print("=" * 70)
    
    for strategy_name, strategy_func in strategies_dict.items():
        stats = simulate(num_games, strategy_func, verbose=False)
        results[strategy_name] = stats['aggregate']
    
    # Display results in a table
    print("\nStrategy Comparison Results:")
    print("=" * 85)
    print(f"{'Strategy':<25} {'Average Score':<18} {'Std Deviation':<15} {'Win %':<10}")
    print("-" * 85)
    
    for strategy_name, stats in results.items():
        avg_score = stats['avg_final_score']
        std_dev = stats['std_dev_final_score']
        win_rate = stats['win_rate'] * 100  # Convert to percentage
        print(f"{strategy_name:<25} {avg_score:>15.2f}     {std_dev:>15.2f}     {win_rate:>8.2f}%")
    
    print("=" * 85)
    
    return results


if __name__ == "__main__":
    # Compare all strategies (including AI if trained)
    # NOTE: Run train_rl_ai.py first to train the AI model
    strategies = {
        "50/50": strategy_50_50,
        "Draw Until 1": strategy_draw_until_one,
        "Draw Until 2": strategy_draw_until_two,
        "Draw Until 3": strategy_draw_until_three,
        "Draw Until 4": strategy_draw_until_four,
    }
    
    # Add AI strategy if available (will use globally trained model)
    if _rl_ai_available:
        strategies["AI"] = strategy_AI
        print("Note: AI strategy included (make sure to run train_rl_ai.py first to train the model)")
    else:
        print("Note: AI strategy not available. Run train_rl_ai.py to train the model first.")
    
    compare_strategies(20, strategies)

