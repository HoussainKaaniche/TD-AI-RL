
import random
import matplotlib.pyplot as plt

from game import create_board
from algorithms import minimax, alphabeta, mcts


# 1. Executer un algorithme et recuperer ses statistiques
def benchmark(board, algorithm, **kwargs):
    result = algorithm(board, **kwargs)

    return {
        "algorithm": algorithm.__name__,
        "move": result["move"],
        "score": result["score"],
        "nodes": result["nodes_explored"],
        "time": result["time"]
    }


# 2. Initialiser le jeu
board = create_board()
results = []

depths = [1, 2, 3, 4, 5]
simulations = [100, 500, 1000, 2000]

minimax_nodes = []
alphabeta_nodes = []
mcts_times = []


# 3. Comparer Minimax et Alpha-Beta
for depth in depths:
    result_min = benchmark(board, minimax, depth=depth)
    result_ab = benchmark(board, alphabeta, depth=depth)

    results.append(("Minimax", f"Depth {depth}", result_min))
    results.append(("Alpha-Beta", f"Depth {depth}", result_ab))

    minimax_nodes.append(result_min["nodes"])
    alphabeta_nodes.append(result_ab["nodes"])

    assert result_min["score"] == result_ab["score"]
    assert result_ab["nodes"] <= result_min["nodes"]


# 4. Comparer les simulations MCTS
for n in simulations:
    result = benchmark(board, mcts, num_simulations=n, seed=42)

    results.append(("MCTS", f"{n} simulations", result))
    mcts_times.append(result["time"])


# 5. Afficher le tableau
print(f"{'Algorithm':<15} {'Configuration':<20} {'Move':<8} "
      f"{'Score':<12} {'Nodes':<10} {'Time (s)':<12}")

print("-" * 85)

for name, config, result in results:
    print(f"{name:<15} {config:<20} {str(result['move']):<8} "
          f"{result['score']:<12.4f} {result['nodes']:<10} "
          f"{result['time']:<12.4f}")


# Deux graphiques dans une seule figure
plt.figure(figsize=(14, 5))

# 6. Graphique Minimax vs Alpha-Beta
plt.subplot(1, 2, 1)
plt.plot(depths, minimax_nodes, marker="o", label="Minimax")
plt.plot(depths, alphabeta_nodes, marker="o", label="Alpha-Beta")

plt.xlabel("Search Depth")
plt.ylabel("Nodes Explored")
plt.title("Minimax vs Alpha-Beta")
plt.legend()
plt.grid()


# 7. Graphique MCTS
plt.subplot(1, 2, 2)
plt.plot(simulations, mcts_times, marker="o", label="MCTS")

plt.xlabel("Number of Simulations")
plt.ylabel("Execution Time (s)")
plt.title("MCTS Performance")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("algorithms_comparison.png")
plt.show()
