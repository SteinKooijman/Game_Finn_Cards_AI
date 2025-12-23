import sys
import os


class PlayingCard:
    """Represents a single playing card with a suit and rank."""
    
    SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    RANKS = ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']
    
    def __init__(self, suit, rank):
        """
        Initialize a playing card.
        
        Args:
            suit: The suit of the card (Hearts, Diamonds, Clubs, or Spades)
            rank: The rank of the card (Ace, 2-10, Jack, Queen, or King)
        """
        if suit not in self.SUITS:
            raise ValueError(f"Invalid suit: {suit}. Must be one of {self.SUITS}")
        if rank not in self.RANKS:
            raise ValueError(f"Invalid rank: {rank}. Must be one of {self.RANKS}")
        
        self.suit = suit
        self.rank = rank
    
    def __str__(self):
        """Return string representation of the card."""
        return f"{self.rank} of {self.suit}"
    
    def __repr__(self):
        """Return detailed string representation of the card."""
        return f"PlayingCard('{self.suit}', '{self.rank}')"
    
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


class CardGame:
    """A console card game where you draw cards and try to avoid duplicate suits."""
    
    def __init__(self):
        """Initialize the game with a shuffled deck."""
        self.deck = Deck()
        self.deck.shuffle()
        self.hand = []
        self.score = 0
        self.total_score = 0  # Track total score across all rounds
        self.ace_suits = set()  # Track the suits of Aces that allow unlimited draws
    
    def has_duplicate_suit(self):
        """
        Check if the hand has duplicate suits.
        If an Ace has been drawn, duplicates of that suit are allowed.
        """
        suits = [card.suit for card in self.hand]
        
        # If we have Ace suits, allow duplicates of those suits
        if self.ace_suits:
            # Count suits excluding all Ace suits
            other_suits = [suit for suit in suits if suit not in self.ace_suits]
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
        # If this is an Ace, add its suit to ace_suits BEFORE checking duplicates
        # This allows an Ace to be the second card of a suit without losing
        if card.rank == 'Ace':
            self.ace_suits.add(card.suit)
        
        self.hand.append(card)
        
        # Check for duplicate suits (ace_suits are already excluded in has_duplicate_suit)
        if self.has_duplicate_suit():
            return False, f"You added {card}! You have duplicate suits - you lose this hand!"
        
        ace_msg = ""
        if card.rank == 'Ace':
            ace_msg = f" 🎯 You can now draw unlimited {card.suit} cards!"
        elif card.suit in self.ace_suits:
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
            hand_str += f"  {i}. {card.rank} of {card.suit} (value: {card.get_value()})\n"
        
        suits = [card.suit for card in self.hand]
        hand_str += f"\nSuits in hand: {', '.join(suits)}"
        
        if self.ace_suits:
            ace_suits_list = sorted(list(self.ace_suits))
            if len(ace_suits_list) == 1:
                hand_str += f"\n🎯 Ace of {ace_suits_list[0]} drawn - unlimited {ace_suits_list[0]} cards allowed!"
            else:
                ace_suits_str = ", ".join(ace_suits_list)
                hand_str += f"\n🎯 Aces of {ace_suits_str} drawn - unlimited cards of these suits allowed!"
        
        return hand_str
    
    def display_deck_order(self):
        """Display the current order of cards in the deck (optional, for debugging)."""
        if not self.deck.cards:
            return "Deck is empty."
        
        deck_str = f"Deck order (top to bottom, {len(self.deck.cards)} cards):\n"
        # Deck is stored with top card at the end of the list, so we reverse to show top to bottom
        for i, card in enumerate(reversed(self.deck.cards), 1):
            deck_str += f"  {i}. {card.rank} of {card.suit}\n"
        
        return deck_str
    
    def reset_hand(self):
        """Reset the hand for a new round."""
        self.hand = []
        self.score = 0
        self.ace_suits = set()  # Reset Ace suits for new round
    
    def add_to_total_score(self, points):
        """Add points to the total score."""
        self.total_score += points


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
    print("- When you draw a card, it is automatically added to your hand")
    print("- If you have two cards of the same suit, you lose (0 points)")
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
            
            # Draw a card
            drawn_card, draw_message = game.draw_card()
            
            if drawn_card is None:
                # Deck is empty
                continue
            
            # Automatically add card to hand
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
                game.add_to_total_score(0)
                game.reset_hand()
                round_number += 1
                input("\nPress Enter to continue...")
                continue
            else:
                # Successfully added card - will show updated state on next loop
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

