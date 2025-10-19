import random
#functa pentru experiment,o singura data
def simulate_game():
    starter = random.choice(["P0", "P1"])
    n = random.randint(1, 6)

    if starter == "P0":
        p_heads = 4/7
    else:
        p_heads = 0.5
        
    m = sum(random.random() < p_heads for _ in range(2 * n))

    if n >= m:
        return starter
    else:
        return "P1" if starter == "P0" else "P0"

#Simulam experimentul de 10000 ori
games = [simulate_game() for _ in range(10000)]
from collections import Counter
results = Counter(games)
print("Simulation results:", results)
