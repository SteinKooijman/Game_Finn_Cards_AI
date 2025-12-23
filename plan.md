# Card Game - Simplified Version Plan

## Overview
A simplified console card game where players draw cards and try to avoid duplicate suits. This version removes the "put card back" mechanic, making the game simpler and more straightforward.

## Core Game Mechanics

### Objective
- Draw cards from the deck and add them to your hand
- Avoid having duplicate suits in your hand (except for the Ace suit)
- Retire your hand to score points
- Play until the deck is empty
- Maximize your total score across all rounds

### Rules
1. **Drawing Cards**: Draw one card at a time from the top of the deck
2. **Adding to Hand**: When you draw a card, you MUST add it to your hand (no option to put it back)
3. **Duplicate Suit Rule**: If you have two cards of the same suit in your hand, you lose the round (0 points)
4. **Ace Special Rule**: If you draw an Ace, you can draw unlimited cards of that suit without losing
5. **Retiring Hand**: You can retire your hand at any time to score points
6. **Scoring**: Score = (sum of card values) × (number of cards in hand)
7. **Card Values**: Ace=1, 2-10=face value, Jack=11, Queen=12, King=13

## Classes and Methods

### PlayingCard Class
- `__init__(suit, rank)`: Initialize a card (no face_up parameter needed)
- `__str__()`: Return string representation
- `__repr__()`: Return detailed representation
- `__eq__(other)`: Check if two cards are equal
- `__hash__()`: Make card hashable
- `get_value()`: Return numeric value of the card

**Constants:**
- `SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']`
- `RANKS = ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']`

### Deck Class
- `__init__()`: Initialize a standard 52-card deck
- `_create_deck()`: Create all 52 cards
- `__str__()`: Return string representation
- `__repr__()`: Return detailed representation
- `__len__()`: Return number of cards
- `shuffle()`: Shuffle the deck
- `deal_card()`: Remove and return the top card from the deck
- `deal_cards(num_cards)`: Deal multiple cards
- `reset()`: Reset to a full 52-card deck

**Note:** Remove `add_to_bottom()` and `add_cards_on_top()` methods as they're not needed.

### CardGame Class
- `__init__()`: Initialize game with shuffled deck
  - `self.deck`: Deck instance
  - `self.hand`: List of cards in hand
  - `self.score`: Current round score
  - `self.total_score`: Total score across all rounds
  - `self.ace_suit`: Suit of the Ace that allows unlimited draws (None if no Ace drawn)

- `has_duplicate_suit()`: Check if hand has duplicate suits (excluding ace_suit)
  - Returns True if duplicates exist, False otherwise

- `draw_card()`: Draw a card from the deck
  - Returns: `(card: PlayingCard or None, message: str)`
  - If deck is empty, returns (None, error message)

- `add_card_to_hand(card)`: Add a drawn card to the hand
  - Checks for duplicate suits
  - Sets ace_suit if Ace is drawn
  - Returns: `(success: bool, message: str)`
  - If duplicate suits found, returns (False, lose message)

- `retire_hand()`: Retire the hand and calculate score
  - Returns: `int` (score for the hand)
  - Score = sum of card values × number of cards

- `display_hand()`: Display current hand
  - Shows all cards with their values
  - Shows suits in hand
  - Shows Ace suit special status if applicable

- `display_deck_order()`: Display current deck order (optional, for debugging)
  - Shows cards remaining in deck

- `reset_hand()`: Reset hand for new round
  - Clears hand, resets score, resets ace_suit

- `add_to_total_score(points)`: Add points to total score

**Removed Methods:**
- `put_drawn_card_back()`: Not needed in simplified version
- `put_card_back_to_bottom()`: Not needed in simplified version

## Game Flow

### Main Game Loop (`play_game()` function)

1. **Initialization**
   - Create CardGame instance
   - Initialize round_number = 1
   - Display welcome message and rules
   - Wait for user to press Enter

2. **Main Loop** (while deck has cards OR hand has cards)
   - Clear screen
   - Display round number, cards remaining, total score
   - Display current hand (or "Your hand is empty")
   
   - **Check game end conditions:**
     - If deck is empty AND hand is empty → Game Over
     - If deck is empty BUT hand has cards → Only allow "Retire hand"
   
   - **Display menu:**
     - If deck has cards: "1. Draw a card", "2. Retire your hand"
     - If deck is empty: "2. Retire your hand" (only option)
   
   - **Handle user choice:**
     
     **Choice 1: Draw a card**
     - Draw card from deck
     - If deck is empty, continue loop
     - Display drawn card
     - **Automatically add card to hand** (no choice given)
     - Check if adding card causes duplicate suits:
       - If yes: Display lose message, reset hand, increment round
       - If no: Continue loop (card is now in hand)
     
     **Choice 2: Retire your hand**
     - If hand is empty and deck is empty → Game Over
     - If hand is empty but deck has cards → Continue loop
     - Calculate score
     - Display retirement message with score breakdown
     - Add score to total score
     - Reset hand
     - Increment round number
     - Wait for user to press Enter
     - Continue loop
     
     **Invalid choice:**
     - Continue loop (will redisplay menu)

3. **Game Over**
   - Display final total score
   - Display total rounds played
   - Thank player

## User Interface

### Welcome Screen
```
==================================================
Welcome to the Card Game!
==================================================

Rules:
- Draw cards from the deck
- When you draw a card, it is automatically added to your hand
- If you have two cards of the same suit, you lose (0 points)
- SPECIAL: If you draw an Ace, you can draw unlimited cards of that suit!
- Retire your hand to score points
- Score = (sum of card values) × (number of cards)
- Ace=1, 2-10=face value, Jack=11, Queen=12, King=13
- Play until the deck is empty!
==================================================
```

### Main Game Screen
```
==================================================
Round 1
==================================================
Cards remaining in deck: 50
Total score so far: 0

Your hand:
  1. 2 of Clubs (value: 2)
  2. 7 of Hearts (value: 7)

Suits in hand: Clubs, Hearts
🎯 Ace of Spades drawn - unlimited Spades cards allowed!

What would you like to do?
1. Draw a card
2. Retire your hand

Enter your choice (1-2):
```

### Card Draw Screen
```
==================================================
Round 1
==================================================
Cards remaining in deck: 49
Total score so far: 0

Your hand:
  1. 2 of Clubs (value: 2)

Suits in hand: Clubs

You drew: King of Hearts

[Card is automatically added to hand]
```

### Lose Screen
```
==================================================
Round 1 - LOST!
==================================================

You added King of Hearts! You have duplicate suits - you lose this hand!

Final hand:
  1. 2 of Clubs (value: 2)
  2. 7 of Clubs (value: 7)
  3. King of Hearts (value: 13)

Suits in hand: Clubs, Clubs, Hearts

Score for this round: 0
Better luck next round!

Press Enter to continue...
```

### Retire Screen
```
==================================================
Round 1 - RETIRED!
==================================================

Your hand:
  1. 2 of Clubs (value: 2)
  2. 7 of Hearts (value: 7)
  3. King of Diamonds (value: 13)

Suits in hand: Clubs, Hearts, Diamonds

You retired your hand!
Sum of card values: 22
Number of cards: 3
Round score: 22 × 3 = 66

Total score: 66

Press Enter to continue...
```

### Game Over Screen
```
==================================================
Game Over!
==================================================
Final total score: 150
Total rounds played: 3

Thanks for playing!
```

## Implementation Details

### Key Simplifications
1. **No Face-Up/Face-Down States**: All cards are treated the same (no `face_up` attribute needed)
2. **No Put Back Option**: When you draw a card, it's automatically added to your hand
3. **Simpler Game Loop**: No branching for "put back" action
4. **No Deck Manipulation**: No need to add cards to bottom or top of deck

### Code Structure
```
card.py
├── PlayingCard class
│   ├── __init__(suit, rank)
│   ├── get_value()
│   └── [other standard methods]
├── Deck class
│   ├── __init__()
│   ├── shuffle()
│   ├── deal_card()
│   └── [other standard methods]
├── CardGame class
│   ├── __init__()
│   ├── has_duplicate_suit()
│   ├── draw_card()
│   ├── add_card_to_hand(card)
│   ├── retire_hand()
│   ├── display_hand()
│   ├── reset_hand()
│   └── add_to_total_score(points)
└── play_game() function
    └── Main game loop
```

### Edge Cases to Handle
1. **Empty Deck**: Check if deck is empty before drawing
2. **Empty Hand**: Don't allow retiring empty hand (unless deck is also empty)
3. **Ace Suit**: Track which suit allows duplicates
4. **Multiple Aces**: Only the first Ace sets the ace_suit (subsequent Aces of different suits don't change it)
5. **Invalid Input**: Handle invalid menu choices gracefully

## Testing Checklist
- [ ] Draw card and add to hand automatically
- [ ] Duplicate suit detection works correctly
- [ ] Ace special rule allows unlimited cards of that suit
- [ ] Scoring calculation is correct
- [ ] Hand retirement works
- [ ] Round progression works
- [ ] Game over when deck and hand are empty
- [ ] Cannot retire empty hand (unless deck is also empty)
- [ ] Total score accumulates correctly
- [ ] Display functions show correct information

## Differences from Full Version
1. **Removed**: `put_drawn_card_back()` method
2. **Removed**: `put_card_back_to_bottom()` method
3. **Removed**: `face_up` attribute from PlayingCard
4. **Removed**: `add_to_bottom()` and `add_cards_on_top()` from Deck
5. **Simplified**: Game loop - no "put back" option in menu
6. **Simplified**: When drawing a card, automatically add to hand (no user choice)
7. **Simplified**: No face-up card handling logic

