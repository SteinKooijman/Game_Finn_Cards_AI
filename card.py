import sys
import os


class PlayingCard:
    """Represents a single playing card with a suit and rank."""
    
    SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    RANKS = ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']
    
    def __init__(self, suit, rank, face_up=False):
        """
        Initialize a playing card.
        
        Args:
            suit: The suit of the card (Hearts, Diamonds, Clubs, or Spades)
            rank: The rank of the card (Ace, 2-10, Jack, Queen, or King)
            face_up: Whether the card is face up (default False)
        """
        if suit not in self.SUITS:
            raise ValueError(f"Invalid suit: {suit}. Must be one of {self.SUITS}")
        if rank not in self.RANKS:
            raise ValueError(f"Invalid rank: {rank}. Must be one of {self.RANKS}")
        
        self.suit = suit
        self.rank = rank
        self.face_up = face_up
    
    def __str__(self):
        """Return string representation of the card."""
        face_up_str = " [FACE UP]" if self.face_up else ""
        return f"{self.rank} of {self.suit}{face_up_str}"
    
    def __repr__(self):
        """Return detailed string representation of the card."""
        return f"PlayingCard('{self.suit}', '{self.rank}', face_up={self.face_up})"
    
    def __eq__(self, other):
        """Check if two cards are equal."""
        if not isinstance(other, PlayingCard):
            return False
        return self.suit == other.suit and self.rank == other.rank
    
    def __hash__(self):
        """Make card hashable for use in sets/dictionaries."""
        return hash((self.suit, self.rank))
    
    def get_value(self):
        """
        Get the numeric value of the card.
        
        Returns:
            int: Ace=1, 2-10=face value, Jack=11, Queen=12, King=13
        """
        rank_values = {
            'Ace': 1,
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
            '7': 7, '8': 8, '9': 9, '10': 10,
            'Jack': 11, 'Queen': 12, 'King': 13
        }
        return rank_values[self.rank]


class Deck:
    """Represents a deck of playing cards."""
    
    def __init__(self):
        """Initialize a standard 52-card deck."""
        self.cards = []
        self._create_deck()
    
    def _create_deck(self):
        """Create a standard 52-card deck with all suits and ranks."""
        self.cards = [
            PlayingCard(suit, rank)
            for suit in PlayingCard.SUITS
            for rank in PlayingCard.RANKS
        ]
    
    def __str__(self):
        """Return string representation of the deck."""
        return f"Deck with {len(self.cards)} cards"
    
    def __repr__(self):
        """Return detailed string representation of the deck."""
        return f"Deck({len(self.cards)} cards)"
    
    def __len__(self):
        """Return the number of cards in the deck."""
        return len(self.cards)
    
    def shuffle(self):
        """Shuffle the deck using random.shuffle."""
        import random
        random.shuffle(self.cards)
    
    def deal_card(self):
        """Remove and return the top card from the deck."""
        if not self.cards:
            raise ValueError("Cannot deal from an empty deck")
        return self.cards.pop()
    
    def deal_cards(self, num_cards):
        """Deal a specified number of cards from the deck."""
        if num_cards > len(self.cards):
            raise ValueError(f"Cannot deal {num_cards} cards from a deck with {len(self.cards)} cards")
        return [self.cards.pop() for _ in range(num_cards)]
    
    def reset(self):
        """Reset the deck to a full 52-card deck."""
        self._create_deck()
    
    def add_to_bottom(self, card):
        """Add a card to the bottom of the deck."""
        self.cards.insert(0, card)
    
    def add_cards_on_top(self, cards):
        """Add multiple cards on top of the deck (in order)."""
        # Since deal_card() uses pop() (removes from end), append cards in order
        # so first card in list is drawn first next time
        for card in cards:
            self.cards.append(card)


class CardGame:
    """A console card game where you draw cards and try to avoid duplicate suits."""
    
    def __init__(self):
        """Initialize the game with a shuffled deck."""
        self.deck = Deck()
        self.deck.shuffle()
        self.hand = []
        self.score = 0
        self.total_score = 0  # Track total score across all rounds
        self.ace_suit = None  # Track the suit of the Ace that allows unlimited draws
    
    def has_duplicate_suit(self):
        """
        Check if the hand has duplicate suits.
        If an Ace has been drawn, duplicates of that suit are allowed.
        """
        suits = [card.suit for card in self.hand]
        
        # If we have an Ace suit, allow duplicates of that suit
        if self.ace_suit:
            # Count suits excluding the Ace suit
            other_suits = [suit for suit in suits if suit != self.ace_suit]
            return len(other_suits) != len(set(other_suits))
        
        # Normal duplicate check
        return len(suits) != len(set(suits))
    
    def draw_card(self):
        """
        Draw a card from the deck (does NOT add to hand).
        
        Returns:
            tuple: (card: PlayingCard or None, message: str)
        """
        if not self.deck.cards:
            return None, "Deck is empty! Cannot draw more cards."
        
        card = self.deck.deal_card()
        return card, f"You drew: {card}"
    
    def add_card_to_hand(self, card):
        """
        Add a drawn card to the hand and check for duplicate suits.
        If an Ace is drawn, you can draw unlimited cards of that suit.
        
        Args:
            card: The card to add to the hand
            
        Returns:
            tuple: (success: bool, message: str)
        """
        self.hand.append(card)
        
        # Check if this is an Ace - if so, set the ace_suit
        if card.rank == 'Ace' and self.ace_suit is None:
            self.ace_suit = card.suit
        
        if self.has_duplicate_suit():
            return False, f"You added {card}! You have duplicate suits - you lose this hand!"
        
        ace_msg = ""
        if card.rank == 'Ace' and self.ace_suit == card.suit:
            ace_msg = f" 🎯 You can now draw unlimited {card.suit} cards!"
        elif self.ace_suit and card.suit == self.ace_suit:
            ace_msg = f" ✓ {card.suit} card (Ace suit - allowed)"
        
        return True, f"Added {card} to your hand.{ace_msg}"
    
    def retire_hand(self):
        """
        Retire the hand and calculate the score.
        
        Returns:
            int: The score for the hand (sum of card values × number of cards)
        """
        if not self.hand:
            return 0
        
        total_value = sum(card.get_value() for card in self.hand)
        num_cards = len(self.hand)
        self.score = total_value * num_cards
        return self.score
    
    def display_hand(self):
        """Display the current hand."""
        if not self.hand:
            return "Your hand is empty."
        
        hand_str = "Your hand:\n"
        for i, card in enumerate(self.hand, 1):
            face_up_indicator = " [FACE UP]" if card.face_up else ""
            hand_str += f"  {i}. {card.rank} of {card.suit}{face_up_indicator} (value: {card.get_value()})\n"
        
        suits = [card.suit for card in self.hand]
        hand_str += f"\nSuits in hand: {', '.join(suits)}"
        
        if self.ace_suit:
            hand_str += f"\n🎯 Ace of {self.ace_suit} drawn - unlimited {self.ace_suit} cards allowed!"
        
        return hand_str
    
    def display_deck_order(self):
        """Display the current order of cards in the deck with their face-up/face-down states."""
        if not self.deck.cards:
            return "Deck is empty."
        
        deck_str = f"Deck order (top to bottom, {len(self.deck.cards)} cards):\n"
        # Deck is stored with top card at the end of the list, so we reverse to show top to bottom
        for i, card in enumerate(reversed(self.deck.cards), 1):
            face_up_indicator = " [FACE UP]" if card.face_up else " [face down]"
            deck_str += f"  {i}. {card.rank} of {card.suit}{face_up_indicator}\n"
        
        return deck_str
    
    def reset_hand(self):
        """Reset the hand for a new round."""
        self.hand = []
        self.score = 0
        self.ace_suit = None  # Reset Ace suit for new round
    
    def add_to_total_score(self, points):
        """Add points to the total score."""
        self.total_score += points
    
    def put_drawn_card_back(self, card):
        """
        Put a drawn card (not yet in hand) back to the bottom of the deck, 
        mark it as face up, and place cards on top based on the card's value.
        
        Args:
            card: The card to put back to the bottom (not in hand)
            
        Returns:
            tuple: (success: bool, message: str, lost: bool, face_up_cards: list)
            - success: Whether the operation completed successfully
            - message: Description of what happened
            - lost: Whether the player lost due to duplicate suits
            - face_up_cards: List of face-up cards that were automatically added to hand
        """
        if card.face_up:
            return False, "Cannot put face-up card back - must be added to hand!", False, []
        
        if not self.deck.cards:
            return False, "Cannot put card back - deck is empty!", False, []
        
        # Mark card as face up
        card.face_up = True
        
        # Determine how many cards to draw from bottom (card value, but limited by available cards)
        num_cards_to_draw = min(card.get_value(), len(self.deck.cards))
        
        if num_cards_to_draw == 0:
            # No cards to draw, just add to bottom
            self.deck.add_to_bottom(card)
            return True, f"Put {card.rank} of {card.suit} back to bottom (face up). No cards to place on top.", False, []
        
        # Draw cards from the bottom BEFORE adding our card (from index 0, 1, 2...)
        drawn_cards = []
        face_up_cards_drawn = []
        
        # Draw from the bottom (index 0, 1, 2...)
        for i in range(num_cards_to_draw):
            if len(self.deck.cards) == 0:
                break
            # Draw from position 0 (the bottom of the deck)
            drawn_card = self.deck.cards.pop(0)
            drawn_cards.append(drawn_card)
            
            # If face up, add to hand immediately and check for duplicate suits
            if drawn_card.face_up:
                face_up_cards_drawn.append(drawn_card)
                self.hand.append(drawn_card)
                
                # Check if this is an Ace - if so, set the ace_suit
                if drawn_card.rank == 'Ace' and self.ace_suit is None:
                    self.ace_suit = drawn_card.suit
                
                # Check for duplicate suits after each face up card addition
                if self.has_duplicate_suit():
                    # Put remaining non-face-up cards back on top of the deck
                    remaining_cards = [c for c in drawn_cards if not c.face_up]
                    if remaining_cards:
                        self.deck.add_cards_on_top(remaining_cards)
                    # Add our card to bottom before returning
                    self.deck.add_to_bottom(card)
                    return False, f"Put {card.rank} of {card.suit} back to bottom. Drew face up card {drawn_card} which caused duplicate suits - you lose!", True, face_up_cards_drawn
        
        # Add our card to the bottom of the deck
        self.deck.add_to_bottom(card)
        
        # Place non-face-up cards on top of the deck
        non_face_up_cards = [c for c in drawn_cards if not c.face_up]
        if non_face_up_cards:
            self.deck.add_cards_on_top(non_face_up_cards)
        
        # Build message
        face_up_msg = ""
        if face_up_cards_drawn:
            face_up_msg = f" Face up cards drawn: {', '.join(str(c) for c in face_up_cards_drawn)}"
        
        return True, f"Put {card.rank} of {card.suit} back to bottom (face up). Placed {len(non_face_up_cards)} card(s) on top.{face_up_msg}", False, face_up_cards_drawn
    
    def put_card_back_to_bottom(self, card):
        """
        Put a card back to the bottom of the deck, mark it as face up,
        and place cards on top based on the card's value.
        
        Args:
            card: The card to put back to the bottom
            
        Returns:
            tuple: (success: bool, message: str, lost: bool)
            - success: Whether the operation completed successfully
            - message: Description of what happened
            - lost: Whether the player lost due to duplicate suits
        """
        if not self.deck.cards:
            return False, "Cannot put card back - deck is empty!", False
        
        # Mark card as face up
        card.face_up = True
        
        # Determine how many cards to draw from bottom (card value, but limited by available cards)
        num_cards_to_draw = min(card.get_value(), len(self.deck.cards))
        
        if num_cards_to_draw == 0:
            # No cards to draw, just add to bottom
            self.deck.add_to_bottom(card)
            return True, f"Put {card.rank} of {card.suit} back to bottom (face up). No cards to place on top.", False
        
        # Draw cards from the bottom BEFORE adding our card (from index 0, 1, 2...)
        drawn_cards = []
        face_up_cards_drawn = []
        
        # Draw from the bottom (index 0, 1, 2...)
        for i in range(num_cards_to_draw):
            if len(self.deck.cards) == 0:
                break
            # Draw from position 0 (the bottom of the deck)
            drawn_card = self.deck.cards.pop(0)
            drawn_cards.append(drawn_card)
            
            # If face up, add to hand immediately and check for duplicate suits
            if drawn_card.face_up:
                face_up_cards_drawn.append(drawn_card)
                self.hand.append(drawn_card)
                
                # Check if this is an Ace - if so, set the ace_suit
                if drawn_card.rank == 'Ace' and self.ace_suit is None:
                    self.ace_suit = drawn_card.suit
                
                # Check for duplicate suits after each face up card addition
                if self.has_duplicate_suit():
                    # Put remaining non-face-up cards back on top of the deck
                    remaining_cards = [c for c in drawn_cards if not c.face_up]
                    if remaining_cards:
                        self.deck.add_cards_on_top(remaining_cards)
                    # Add our card to bottom before returning
                    self.deck.add_to_bottom(card)
                    return False, f"Put {card.rank} of {card.suit} back to bottom. Drew face up card {drawn_card} which caused duplicate suits - you lose!", True
        
        # Add our card to the bottom of the deck
        self.deck.add_to_bottom(card)
        
        # Place non-face-up cards on top of the deck
        non_face_up_cards = [c for c in drawn_cards if not c.face_up]
        if non_face_up_cards:
            self.deck.add_cards_on_top(non_face_up_cards)
        
        # Build message
        face_up_msg = ""
        if face_up_cards_drawn:
            face_up_msg = f" Face up cards drawn: {', '.join(str(c) for c in face_up_cards_drawn)}"
        
        return True, f"Put {card.rank} of {card.suit} back to bottom (face up). Placed {len(non_face_up_cards)} card(s) on top.{face_up_msg}", False


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def play_game():
    """Main game loop."""
    game = CardGame()
    
    round_number = 1
    
    # Show welcome message once at the start
    print("=" * 50)
    print("Welcome to the Card Game!")
    print("=" * 50)
    print("\nRules:")
    print("- Draw cards from the deck")
    print("- After drawing, choose to add the card to your hand or put it back")
    print("- Face-up cards must be added to your hand (cannot be put back)")
    print("- If you add two cards of the same suit, you lose (0 points)")
    print("- SPECIAL: If you draw an Ace, you can draw unlimited cards of that suit!")
    print("- Retire your hand to score points")
    print("- Score = (sum of card values) × (number of cards)")
    print("- Ace=1, 2-10=face value, Jack=11, Queen=12, King=13")
    print("- Play until the deck is empty!")
    print("=" * 50)
    input("\nPress Enter to start...")
    clear_screen()
    
    while len(game.deck) > 0 or game.hand:
        clear_screen()
        print("=" * 50)
        print(f"Round {round_number}")
        print("=" * 50)
        print(f"Cards remaining in deck: {len(game.deck)}")
        print(f"Total score so far: {game.total_score}")
        
        if game.hand:
            print("\n" + game.display_hand())
        else:
            print("\nYour hand is empty.")
        
        # Check if deck is empty and hand is empty - game over
        if len(game.deck) == 0 and not game.hand:
            break
        
        # Check if deck is empty but hand has cards - can only retire
        if len(game.deck) == 0:
            print("\n⚠️  Deck is empty! You can only retire your hand now.")
            print("\nWhat would you like to do?")
            print("2. Retire your hand")
            sys.stdout.flush()
            choice = input("\nEnter your choice (2): ").strip()
        else:
            print("\nWhat would you like to do?")
            print("1. Draw a card")
            print("2. Retire your hand")
            sys.stdout.flush()
            choice = input("\nEnter your choice (1-2): ").strip()
        
        if choice == "1":
            if len(game.deck) == 0:
                continue
            
            # Step 1: Draw a card (not added to hand yet)
            drawn_card, draw_message = game.draw_card()
            
            if drawn_card is None:
                # Deck is empty
                continue
            
            # Step 2: Show the card and ask what to do with it
            clear_screen()
            print("=" * 50)
            print(f"Round {round_number}")
            print("=" * 50)
            print(f"Cards remaining in deck: {len(game.deck)}")
            print(f"Total score so far: {game.total_score}")
            
            if game.hand:
                print("\n" + game.display_hand())
            else:
                print("\nYour hand is empty.")
            
            print(f"\n{draw_message}")
            if drawn_card.face_up:
                print("⚠️  This card is FACE UP - it must be added to your hand!")
            
            # If face up, must add to hand
            if drawn_card.face_up:
                action_choice = "1"
            else:
                print("\nWhat would you like to do with this card?")
                print("1. Add to hand")
                print("2. Put back to bottom of deck")
                sys.stdout.flush()
                action_choice = input("\nEnter your choice (1-2): ").strip()
            
            if action_choice == "1":
                # Add card to hand
                success, message = game.add_card_to_hand(drawn_card)
                
                if not success:
                    # Lost due to duplicate suit
                    clear_screen()
                    print("=" * 50)
                    print(f"Round {round_number} - LOST!")
                    print("=" * 50)
                    print(f"\n{message}")
                    print(f"\nFinal hand:")
                    print(game.display_hand())
                    print(f"\nScore for this round: 0")
                    print("Better luck next round!")
                    print(f"\n{game.display_deck_order()}")
                    game.add_to_total_score(0)
                    game.reset_hand()
                    round_number += 1
                    input("\nPress Enter to continue...")
                    continue
                else:
                    # Successfully added card - will show updated state on next loop
                    continue
            
            elif action_choice == "2":
                # Put card back to bottom
                success, message, lost, face_up_cards = game.put_drawn_card_back(drawn_card)
                
                if not success:
                    # Error putting card back (shouldn't happen for non-face-up cards)
                    continue
                
                if lost:
                    # Lost due to duplicate suit from face up cards
                    clear_screen()
                    print("=" * 50)
                    print(f"Round {round_number} - LOST!")
                    print("=" * 50)
                    print(f"\n{message}")
                    print(f"\nFinal hand:")
                    print(game.display_hand())
                    print(f"\nScore for this round: 0")
                    print("Better luck next round!")
                    print(f"\n{game.display_deck_order()}")
                    game.add_to_total_score(0)
                    game.reset_hand()
                    round_number += 1
                    input("\nPress Enter to continue...")
                    continue
                else:
                    # Successfully put card back - show deck order
                    clear_screen()
                    print("=" * 50)
                    print(f"Round {round_number}")
                    print("=" * 50)
                    print(f"\n{message}")
                    print(f"\n{game.display_deck_order()}")
                    input("\nPress Enter to continue...")
                    continue
            else:
                # Invalid choice - will show error and redisplay on next loop
                continue
        
        elif choice == "2":
            if not game.hand:
                if len(game.deck) == 0:
                    break
                continue
            
            score = game.retire_hand()
            total_value = sum(card.get_value() for card in game.hand)
            num_cards = len(game.hand)
            
            clear_screen()
            print("=" * 50)
            print(f"Round {round_number} - RETIRED!")
            print("=" * 50)
            print(f"\n{game.display_hand()}")
            print(f"\nYou retired your hand!")
            print(f"Sum of card values: {total_value}")
            print(f"Number of cards: {num_cards}")
            print(f"Round score: {total_value} × {num_cards} = {score}")
            
            game.add_to_total_score(score)
            print(f"\nTotal score: {game.total_score}")
            print(f"\n{game.display_deck_order()}")
            
            game.reset_hand()
            round_number += 1
            input("\nPress Enter to continue...")
            continue
        
        else:
            # Invalid choice - will show error and redisplay on next loop
            continue
    
    # Game over message
    clear_screen()
    print("=" * 50)
    print("Game Over!")
    print("=" * 50)
    print(f"Final total score: {game.total_score}")
    print(f"Total rounds played: {round_number - 1}")
    print("\nThanks for playing!")


# Instantiate a classic 52-card deck
if __name__ == "__main__":
    play_game()

