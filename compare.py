import numpy as np
import matplotlib.pyplot as plt
from env import GridWorld
from train import train_qlearning, train_sarsa

def smooth(x, w=50):
    return np.convolve(x, np.ones(w)/w, mode='valid')

def run_comparison(n_runs=30, episodes=1500):
    env = GridWorld(size=10)
    final_ql, final_sa = [], []
    
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}")
        _, r_ql = train_qlearning(env, n_episodes=episodes)
        _, r_sa = train_sarsa(env, n_episodes=episodes)
        final_ql.append(np.mean(r_ql[-100:]))
        final_sa.append(np.mean(r_sa[-100:]))
    
    print(f"\nQ-Learning : {np.mean(final_ql):.2f} ± {np.std(final_ql):.2f}")
    print(f"SARSA      : {np.mean(final_sa):.2f} ± {np.std(final_sa):.2f}")
    
    plt.figure(figsize=(8, 5))
    plt.boxplot([final_ql, final_sa], labels=["Q-Learning", "SARSA"])
    plt.ylabel("Récompense moyenne (100 derniers épisodes)")
    plt.title(f"Comparaison sur {n_runs} runs")
    plt.grid(alpha=0.3)
    plt.savefig("boxplot.png", dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_comparison()