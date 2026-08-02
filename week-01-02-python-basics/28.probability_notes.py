"""
Probability Notes
-----------------
- Probability: Likelihood of an event (0 ≤ P ≤ 1).
- Coin Toss: P(Heads) = 0.5 for a fair coin.
- Dice Roll: P(rolling a 4) = 1/6 for a fair die.
- Simulations: Use random module to simulate events.
"""
import random

def simulate_coin_tosses(n=10):
    """

    :param n: Default value = 10)

    """
    return [random.choice(['H', 'T']) for _ in range(n)]

def simulate_dice_rolls(n=10):
    """

    :param n: Default value = 10)

    """
    return [random.randint(1, 6) for _ in range(n)]

# Example usage
if __name__ == "__main__":
    print("Coin tosses:", simulate_coin_tosses())
    print("Dice rolls:", simulate_dice_rolls())
